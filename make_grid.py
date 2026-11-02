import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
C = "/tmp/claude-1000/-home-euiyul/b2cd91ee-ceb0-4d80-8cdc-476fc73249d3/scratchpad"
panels = [
    (C + "/cmp/scifact_bm25_distribution.png", "BM25", "ROC-AUC 0.849 · best F1 0.268"),
    (C + "/cmp/scifact_dense_distribution.png", "Dense (pretrained)", "ROC-AUC 0.899 · best F1 0.413"),
    (C + "/emb/ft_embedding_distribution.png", "Dense (fine-tuned, section 16)", "ROC-AUC 0.959"),
    (C + "/ce_pre_dist.png", "Cross-Encoder (pretrained)", "ROC-AUC 0.941 · best F1 0.469"),
    (C + "/cmp/crossencoder_finetuned_distribution.png", "Cross-Encoder (fine-tuned)", "ROC-AUC 0.972 · best F1 0.597"),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
for ax in axes.flat:
    ax.axis("off")
for ax, (p, t, sub) in zip(axes.flat, panels):
    ax.imshow(mpimg.imread(p))
    ax.set_title(f"{t}\n{sub}", fontsize=13)
fig.suptitle("SciFact: relevant vs non-relevant score distribution by method", fontsize=16)
fig.tight_layout()
fig.savefig(C + "/cmp/method_comparison_distribution.png", dpi=110)
