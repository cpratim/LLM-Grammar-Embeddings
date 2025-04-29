import torch
import numpy as np
from util.model import clear_cuda_cache, get_device_memory_report, load_model
from module.blocked_attention import (
    load_altered_attention_model,
    disable_head,
    enable_head,
    get_model_dimensions,
)
from util.chat import load_embeddings_dataset, load_embeddings_dataset_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from random import sample
import json


def metric(
    data,
    model,
    tokenizer,
    device,
    metric_type: str,
    grammar: bool,
    use_system_message: bool = False,
):
    embeddings = load_embeddings_dataset(
        data,
        model,
        tokenizer,
        device,
        types_=[metric_type],
        use_tqdm=False,
        grammar=grammar,
        use_system_message=use_system_message,
    )
    embeddings = embeddings[metric_type]
    X = np.array([embedding["states"][-1] for embedding in embeddings])
    y = np.array([embedding["answer"] for embedding in embeddings])
    # print(X[0])

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
    mini_batch_size=50,
    n_iterations=10,
    metric_type="mcq",
    grammar=True,
    outfile="activated.json",
    use_system_message=False,
):

    mini_batch = sample(data, mini_batch_size)
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)

    print(
        f"Starting accuracy (full model): {metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)}"
    )

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            disable_head(model, layer_idx, head_idx)

    # for layer_idx in range(n_layers):
    #     for head_idx in range(n_heads):
    #         enable_head(model, layer_idx, head_idx)

    mini_batch = sample(data, mini_batch_size)
    last_acc = metric(
        mini_batch,
        model,
        tokenizer,
        device,
        metric_type=metric_type,
        grammar=grammar,
        use_system_message=use_system_message,
    )
    print(f"Starting accuracy (disabled all heads): {last_acc}")

    activated = []
    for _ in range(n_iterations):
        print(f"[{_ + 1}/{n_iterations}] | Accuracy: {last_acc:.4f}")
        best_layer_gain, best_layer = last_acc, None
        for layer_idx in tqdm(range(n_layers), desc="Scoring layers"):
            for head_idx in range(n_heads):
                enable_head(model, layer_idx, head_idx)
            score = metric(
                mini_batch,
                model,
                tokenizer,
                device,
                metric_type=metric_type,
                grammar=grammar,
                use_system_message=use_system_message,
            )
            if score > best_layer_gain:
                best_layer_gain = score
                best_layer = layer_idx
            for head_idx in range(n_heads):
                if (layer_idx, head_idx) not in activated:
                    disable_head(model, layer_idx, head_idx)
        print(
            f"[{_ + 1}/{n_iterations}] | Best layer: {best_layer}, gain: {best_layer_gain}"
        )
        if best_layer is None:
            break
        best_head_gain, best_head = last_acc, None
        for head_idx in tqdm(range(n_heads), desc="Scoring heads"):
            if (best_layer, head_idx) in activated:
                continue
            enable_head(model, best_layer, head_idx)
            score = metric(
                mini_batch,
                model,
                tokenizer,
                device,
                metric_type=metric_type,
                grammar=grammar,
                use_system_message=use_system_message,
            )
            if score > best_head_gain:
                best_head_gain = score
                best_head = head_idx
            disable_head(model, best_layer, head_idx)
        print(
            f"[{_ + 1}/{n_iterations}] | Best head: {best_head}, gain: {best_head_gain}"
        )
        if best_layer is not None and best_head is not None:
            activated.append((best_layer, best_head))

        last_acc = best_head_gain
        with open(outfile, "w") as f:
            json.dump(activated, f, indent=4)
    return activated


def head_select_batches(
    data,
    model,
    tokenizer,
    device,
    mini_batch_size=50,
    batch_size=0.2,
    n_iterations=20,
    splits_per_iter=5,
    metric_type="raw",
    grammar=True,
    use_system_message=False,
    outfile="activated_heads.json",
):
    
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
    pool = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    batch_size = int(len(pool) * batch_size)
    mini_batch = sample(data, mini_batch_size)

    print(f"Starting accuracy (full model): {metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)}")

    for layer_idx, head_idx in pool:
        disable_head(model, layer_idx, head_idx)

    best_score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
    print(f"Starting accuracy (disabled all heads): {best_score}")

    activated = []  
    for _ in range(n_iterations):
        best_score_iter, best_head_batch = best_score, None
        for _ in tqdm(range(splits_per_iter), desc="Scoring batches"):
            head_batch = sample(pool, batch_size)
            for layer_idx, head_idx in head_batch:
                enable_head(model, layer_idx, head_idx)
            score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
            if score > best_score_iter:
                best_score_iter = score
                best_head_batch = head_batch
            for layer_idx, head_idx in head_batch:
                disable_head(model, layer_idx, head_idx)
        if best_head_batch is None:
            continue

        new_pool = []
        for layer_idx, head_idx in best_head_batch:
            enable_head(model, layer_idx, head_idx)
            for l, h in pool:
                if (l, h) not in best_head_batch:
                    new_pool.append((l, h))

        pool = new_pool
        best_score = best_score_iter
        print(f"[{_ + 1}/{n_iterations}] | Accuracy: {best_score}")
        activated.extend(best_head_batch)

        with open(outfile, "w") as f:
            json.dump(activated, f, indent=4)


def head_select_layers(
    data,
    model,
    tokenizer,
    device,
    mini_batch_size=50,
    n_iterations=10,
    n_splits_per_iter=5,
    metric_type="raw",
    grammar=True,
    batch_size=.33,
    use_system_message=False,
    outfile="activated_layers.json",
):
    
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
    layer_pool = [i for i in range(n_layers)]
    batch_size = int(len(layer_pool) * batch_size)
    print(batch_size, len(layer_pool))
    mini_batch = sample(data, mini_batch_size)
    print(f"Starting accuracy (full model): {metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)}")
    for layer_idx in layer_pool:
        for head_idx in range(n_heads):
            disable_head(model, layer_idx, head_idx)
    
    # for layer_idx in layer_pool:
    #     for head_idx in range(n_heads):
    #         enable_head(model, layer_idx, head_idx)

    best_score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
    print(f"Starting accuracy (disabled all heads): {best_score}")
    # return
    activated = []
    for _ in range(n_iterations):
        best_score_iter, best_layer_batch = best_score, None
        for _ in tqdm(range(n_splits_per_iter), desc="Scoring layers"):
            layer_batch = sample(layer_pool, batch_size)
            for layer_idx in layer_batch:
                if layer_idx in activated:
                    continue
                for head_idx in range(n_heads):
                    enable_head(model, layer_idx, head_idx)
            score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
            for layer_idx in layer_batch:
                if layer_idx in activated:
                    continue
                for head_idx in range(n_heads):
                    disable_head(model, layer_idx, head_idx)
            if score > best_score_iter:
                best_score_iter = score
                best_layer_batch = layer_batch
            
        if best_layer_batch is None:
            continue

        best_score = best_score_iter
        activated.extend(best_layer_batch)
        print(f"[{_ + 1}/{n_iterations}] | Accuracy: {best_score}")
        
        with open(outfile, "w") as f:
            json.dump(activated, f, indent=4)
            

def head_select_layers_remove(
    data,
    model,
    tokenizer,
    device,
    mini_batch_size=50,
    n_iterations=10,
    metric_type="raw",
    grammar=True,
    use_system_message=False,
    outfile="activated_layers_remove.json",
):
    n_layers, n_heads = get_model_dimensions(model, tokenizer, device)

    mini_batch = sample(data, mini_batch_size)
    score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
    print(f"Starting accuracy (full model): {score}")
    
    activated = []
    for _ in range(n_iterations):
        best_score_iter, best_layer = score, None
        for layer_idx in tqdm(range(n_layers), desc="Scoring layers"):
            if layer_idx in activated:
                continue
            for head_idx in range(n_heads):
                disable_head(model, layer_idx, head_idx)
            iter_score = metric(mini_batch, model, tokenizer, device, metric_type=metric_type, grammar=grammar, use_system_message=use_system_message)
            print(f"[{_ + 1}/{n_iterations}] | Layer: {layer_idx} | Accuracy: {iter_score}")
            if iter_score < best_score_iter:
                best_score_iter = iter_score
                best_layer = layer_idx
            for head_idx in range(n_heads):
                enable_head(model, layer_idx, head_idx)

        activated.append(best_layer)
        score = best_score_iter
        print(f"[{_ + 1}/{n_iterations}] | Accuracy Drop: {score - best_score_iter}")
        
        with open(outfile, "w") as f:
            json.dump(activated, f, indent=4)
            
            


#     n_layers, n_heads = get_model_dimensions(model, tokenizer, device)
#     print(n_layers, n_heads)
