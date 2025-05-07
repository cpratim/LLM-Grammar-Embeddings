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


def get_splits(layer_pool, percentage):
    n_per_split = int(len(layer_pool) * percentage)
    splits = []
    for i in range(0, len(layer_pool), n_per_split):
        if i + n_per_split > len(layer_pool):
            splits[-1].extend(layer_pool[i:])
        else:
            splits.append(layer_pool[i:i+n_per_split])
    return splits


def metric(
    embeddings,
    metric_type: str,
):
    embeddings = embeddings[metric_type]
    X = np.array([embedding["states"][-1] for embedding in embeddings])
    y = np.array([embedding["answer"] for embedding in embeddings])

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    svm = SVC(kernel="linear")
    svm.fit(X_pca, y)
    acc = accuracy_score(y, svm.predict(X_pca))
    return acc


def head_select(
    data,
    model,
    tokenizer,
    device,
    metric_type="raw",
    problem_type="numerical",
    use_system_message=False,
    percentages=[1/2, 1/3, 1/4, 1/5, 1/8, 1/10, 1/20],
    outfile="removed_layers_numerical.json",
):
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
    layer_pool = [l for l in range(n_layers)]
    results = []
    for percentage in percentages:
        splits = get_splits(layer_pool, percentage)
        split_results = []
        print('Disabling layers: ', splits)
        for split in splits:
            for layer in split:
                for head in range(n_heads):
                    disable_head(model, layer, head)
            print('Disabling layers: ', split)
            embeddings = load_embeddings_dataset(
                data,
                model,
                tokenizer,
                device,
                types_=[metric_type],
                use_tqdm=True,
                problem_type=problem_type,
                use_system_message=use_system_message,
            )
            acc = metric(embeddings, metric_type)
            print(f"Accuracy: {acc}")
            for layer in split:
                for head in range(n_heads):
                    enable_head(model, layer, head)

            split_results.append({
                "split": split,
                "accuracy": acc
            })
        results.append({
            "percentage": percentage,
            "splits": split_results
        })
        with open(outfile, "w") as f:
            json.dump(results, f, indent=4)  
    return acc

def get_entropy(distribution):
    entropy = 0
    for i in range(len(distribution)):
        entropy += distribution[i] * np.log(distribution[i])
    return -entropy

            
def remove_high_entopy_heads(
    data,
    model,
    tokenizer,
    device,
    target_heads={},
    metric_type="raw",
    problem_type="numerical",
    head_batch_size=3,
    percentage_to_remove=0.4,
    use_system_message=False,
    outfile="removed_layers_numerical.json",
):
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)

    head_pool = []
    for layer, heads in target_heads.items():
        for head in heads:
            head_pool.append((layer, head))

    for i in range(10):
        print("ITERATION: ", i)
        print("Head pool: ", head_pool)
        lowest_acc = 1
        lowest_acc_head_batch = None

        for _ in range(5):
            head_batch = sample(head_pool, head_batch_size)
            print('Enabling Head batch: ', head_batch)
            for layer, head in head_pool:
                disable_head(model, layer, head)

            for layer, head in head_batch:
                enable_head(model, layer, head)

            disabled_heads = get_disabled_heads(model)
            print('Disabled heads:')
            for i, layer in enumerate(disabled_heads):
                if len(disabled_heads[layer]) > 0:
                    print(f'Layer {i}:', disabled_heads[layer])
            embeddings = load_embeddings_dataset(data, model, tokenizer, device, types_=[metric_type], problem_type=problem_type, use_system_message=use_system_message)
            acc = metric(embeddings, metric_type)
            print(f"Accuracy: {acc}")
            if acc < lowest_acc:
                lowest_acc = acc
                lowest_acc_head_batch = head_batch

            for layer in range(n_layers):
                for head in range(n_heads):
                    enable_head(model, layer, head)
    
        print("Lowest accuracy: ", lowest_acc)
        print("Lowest accuracy head batch: ", lowest_acc_head_batch)

        new_head_pool = []
        for layer, head in head_pool:
            if (layer, head) not in lowest_acc_head_batch:
                new_head_pool.append((layer, head))
        print("New head pool: ", new_head_pool)
        print("--------------------------------")

        head_pool = new_head_pool

            
            


    
        

        