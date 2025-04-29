import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_all_layers_divergence(
    embeddings, method="pca", divisions=[0, 0.25, 0.5, 0.75, 1]
):
    # n_layers = len(embeddings['good_embeddings'][0])
    n_layers = len(embeddings.values()[0]["states"][0])
    layers_idx = np.array([int((n_layers - 1) * x) for x in divisions])
    layers_idx[0] += 1
    for type_ in embeddings:

        states = np.array([x["states"] for x in embeddings[type_]])
        print(states.shape)

        # for j, layer_idx in enumerate(layers_idx):
        #     good_embeddings_layer = [x[layer_idx] for x in embeddings[type_]['good_embeddings']]
        # bad_embeddings_layer = [x[layer_idx] for x in embeddings['bad_embeddings']]

        # bin_embeddings_layer = [x[layer_idx] for x in embeddings['bin_embeddings']]
        # mcq_embeddings_layer = [x[layer_idx] for x in embeddings['mcq_embeddings']]
        # raw_embeddings_layer = np.concatenate([good_embeddings_layer, bad_embeddings_layer])
        # embeddings_layer = [raw_embeddings_layer, bin_embeddings_layer, mcq_embeddings_layer]
        # labels = [raw_labels, bin_labels, mcq_labels]

        # for i, embedding_layer in enumerate(embeddings_layer):
        #     embedding_layer = np.array(embedding_layer)
        #     if method == 'pca':
        #         pca = PCA(n_components=2)
        #         result = pca.fit_transform(embedding_layer)
        #     elif method == 'tsne':
        #         tsne = TSNE(n_components=2, random_state=42, perplexity=50)
        #         result = tsne.fit_transform(embedding_layer)

        #     sns.scatterplot(
        #         x=result[:, 0],
        #         y=result[:, 1],
        #         hue=labels[i],
        #         palette=sns.color_palette("bright", len(np.unique(labels[i]))),
        #         s=10,
        #         ax=axs[j, i]
        #     )
        #     axs[j, i].set_title(f'Layer {layer_idx}')
        #     axs[j, i].legend(title="Label")

    plt.tight_layout()
    plt.show()
