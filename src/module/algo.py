import torch
import numpy as np
from util.model import clear_cuda_cache, get_device_memory_report, load_model
from module.blocked_attention import (
    load_altered_attention_model,
    disable_head,
    enable_head,
    get_model_dimensions,
    get_disabled_heads
)
from util.chat import load_embeddings_dataset, load_embeddings_dataset_batch, load_attentions_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from random import sample
import json
from util.data import load_numerical_data
from random import shuffle
from pprint import pprint


def metric(
    data,
    model,
    tokenizer,
    device,
    metric_type: str,
    problem_type: str,
    D: int = 10,
):
    embeddings = load_embeddings_dataset(
        data,
        model,
        tokenizer,
        device,
        types_=[metric_type],
        use_tqdm=True,
        problem_type=problem_type,
    )[metric_type]
    X = np.array([embedding["states"][-1] for embedding in embeddings])
    y = np.array([embedding["answer"] for embedding in embeddings])

    pca = PCA(n_components=D)
    pca.fit(X)
    X_pca = pca.transform(X)

    svm = SVC(kernel="linear")
    svm.fit(X_pca, y)
    acc = accuracy_score(y, svm.predict(X_pca))
    return acc


def disable_layers(model, layer_subset, n_heads):
    for layer in layer_subset:
        for head in range(n_heads):
            disable_head(model, layer, head)


def enable_layers(model, layer_subset, n_heads):
    for layer in layer_subset:
        for head in range(n_heads):
            enable_head(model, layer, head)


def layer_select(
    data,
    model,
    tokenizer,
    device,
    metric_type="raw",
    problem_type="numerical",
    k: int = 5,
    D: int = 10,
    outfile="outputs/removed_layers_numerical.json",
):
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
    layer_pool = [l for l in range(n_layers)]
    
    results = {}
    for i in range(len(layer_pool) - k + 1):
        layer_subset = layer_pool[i:i+k]
        disable_layers(model, layer_subset, n_heads)
        acc = metric(data, model, tokenizer, device, metric_type, problem_type, D=D)
        for l in layer_subset:
            if l not in results:
                results[l] = []
            results[l].append(acc)
        enable_layers(model, layer_subset, n_heads)
        
        with open(outfile, "w") as f:
            json.dump(results, f, indent=4)
    return results


def save_head_pool(head_pool, outfile, starting_acc, lowest_acc, i, base_acc):
    head_dict = {}
    for (l, h) in head_pool:
        if l not in head_dict:
            head_dict[l] = []
        head_dict[l].append(h)
    res_dict = {
        "base_acc": base_acc,
        "starting_acc": starting_acc,
        "lowest_acc": lowest_acc,
        "i": i,
        "head_pool": head_dict,
    }
    with open(outfile, "w") as f:
        json.dump(res_dict, f, indent=4)

def get_total_disabled_heads(model):
    disabled_heads = get_disabled_heads(model)
    total_disabled_heads = 0
    for layer in disabled_heads:
        total_disabled_heads += len(disabled_heads[layer])
    return total_disabled_heads


def head_select(
    data,
    model, 
    tokenizer, 
    device, 
    epsilon=.1,
    layer_data=None,
    n_splits=10,
    metric_type="raw",
    problem_type="numerical",
    D: int = 10,
    percentile: int = 75,
    outfile="outputs/gemma_numerical_head_pool.json",
):
    if model is not None:
        n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
    total_heads = n_layers * n_heads

    layer_impact = [max(layer_data[k]) for k in layer_data]
    least_impact = max(layer_impact)
    impacts = [(least_impact - v) for v in layer_impact]
    upper_quartile = np.percentile(impacts, percentile)
    layer_pool = [i for i, v in enumerate(layer_impact) if (least_impact - v) >= upper_quartile]
    head_pool = [(l, h) for l in layer_pool for h in range(n_heads)]
    K = int(len(head_pool) / 2)

    base_acc = metric(data, model, tokenizer, device, metric_type, problem_type, D=D)
    print('Base Acc: ', base_acc)

    for l in layer_pool:
        for h in range(n_heads):
            disable_head(model, l, h)
    
    starting_acc = metric(data, model, tokenizer, device, metric_type, problem_type, D=D)
    print('Starting Acc: ', starting_acc)

    for l in layer_pool:
        for h in range(n_heads):
            enable_head(model, l, h)
    
    i = 0
    while True:
        pool_len = len(head_pool)
        if pool_len < K:
            if K == 1:
                break
            K = int(np.ceil(K / 2))
            print('Reducing K to ', K)
            continue

        pool_percent = pool_len / total_heads
        print(f"Iteration {i+1} | K = {K} | Pool = {pool_len} [{pool_percent:.2f}]")
        lowest_acc, lowest_acc_head_batch = 1, None
        for _ in range(n_splits):
            head_batch = sample(head_pool, K)
            for l, h in head_pool:
                disable_head(model, l, h)
            print('Disabled Heads (1): ', get_total_disabled_heads(model))
            for l, h in head_batch:
                enable_head(model, l, h)
            print('Disabled Heads (2): ', get_total_disabled_heads(model))
            acc = metric(data, model, tokenizer, device, metric_type, problem_type, D=D)
            if acc < lowest_acc:
                lowest_acc = acc
                lowest_acc_head_batch = head_batch
        
        print('Lowest Acc: ', lowest_acc, '| Head Batch: ', lowest_acc_head_batch)
    
        for l in range(n_layers):
            for h in range(n_heads):
                enable_head(model, l, h)

        if lowest_acc > starting_acc + epsilon * base_acc:
            if K == 1:
                break
            K = int(np.ceil(K / 2))
            print('Reducing K to ', K)

        else:
            save_head_pool(head_pool, outfile, starting_acc, lowest_acc, i, base_acc)
            new_head_pool = []
            for l, h in head_pool:
                if (l, h) not in lowest_acc_head_batch:
                    new_head_pool.append((l, h))

            head_pool = new_head_pool
            if K != 1 and K > .9 * len(head_pool):
                K = int(np.ceil(K / 2))
                print('Reducing K to ', K)
                continue

        
        i += 1

