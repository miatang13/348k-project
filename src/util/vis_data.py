import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

# def visualize_clustering(data, labels, output_dir):
#     pca = PCA(n_components=2)
#     reduced_data = pca.fit_transform(data)

#     plt.figure(figsize=(10, 7))
#     unique_labels = set(labels)
#     colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

#     for label, color in zip(unique_labels, colors):
#         if label == -1:
#             # Black used for noise.
#             color = "k"
#             marker = "x"
#         else:
#             marker = "o"
#         class_member_mask = labels == label
#         xy = reduced_data[class_member_mask]
#         plt.scatter(
#             xy[:, 0], xy[:, 1], s=50, c=[color], label=f"Cluster {label}", marker=marker
#         )

#     plt.title("DBSCAN Clustering of Consensus Masks")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.legend()
#     plt.savefig(f"{output_dir}/clustering.png")
#     plt.close()
#     plt.clf()
#     print("Saved clustering plot to clustering.png")


def vis_consensus_masks(consensus_masks, output_dir, no_filtering=False,
                        fig_title="Final Representative Masks", mask_output_name="consensus_mask"):
    num_per_row = 4
    fig, ax = plt.subplots(
        math.ceil(len(consensus_masks) / num_per_row), num_per_row, figsize=(8, 4)
    )
    plt.subplots_adjust(
        hspace=0.05, wspace=0.025
    )  # Adjust these values to bring items closer
    fig.suptitle(fig_title)

    for i in range(len(consensus_masks)):
        mask = consensus_masks[i]
        if len(consensus_masks) <= num_per_row:
            ax[i].imshow(mask, cmap="gray")
        else:
            ax[i // num_per_row, i % num_per_row].imshow(mask, cmap="gray")

    for sub_ax in ax.flat:
        sub_ax.set_yticks([])
        sub_ax.set_xticks([])

    if len(consensus_masks) < num_per_row:
        for i in range(len(consensus_masks), num_per_row):
            ax[i].axis("off")
    else:
        for i in range(
            len(consensus_masks),
            math.ceil(len(consensus_masks) / num_per_row) * num_per_row,
        ):
            ax[i // num_per_row, i % num_per_row].axis("off")

    mask_output_name = mask_output_name
    if no_filtering:
        mask_output_name += "_no_filtering"
    mask_path = os.path.join(output_dir, f"{mask_output_name}.png")
    plt.savefig(mask_path)
    plt.close()
    plt.clf()
    print(f"Saved consensus masks plot to {mask_path}")
    return mask_path
