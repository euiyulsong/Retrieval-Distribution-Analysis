import numpy as np, ir_datasets, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
OUT = "/tmp/claude-1000/-home-euiyul/b2cd91ee-ceb0-4d80-8cdc-476fc73249d3/scratchpad/emb"
K = 30; MAX_Q = 150
te = ir_datasets.load("beir/scifact/test")
docs_ds = ir_datasets.load("beir/scifact/train")
docs = {str(d.doc_id): (((getattr(d,"title","") or "")+" "+(getattr(d,"text","") or "")).strip()) for d in docs_ds.docs_iter()}
te_q = {str(q.query_id): q.text for q in te.queries_iter()}
qr = defaultdict(set)
for r in te.qrels_iter():
    if r.relevance > 0: qr[str(r.query_id)].add(str(r.doc_id))
qids = [q for q in qr if q in te_q][:MAX_Q]
import random
random.seed(42)
need = set().union(*[qr[q] for q in qids])
eval_docs = sorted(need | set(random.Random(42).sample(list(docs), 300)))
m = SentenceTransformer(OUT + "/ft_model")
qm = m.encode([te_q[q] for q in qids], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
dm = m.encode([docs[d] for d in eval_docs], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
S = qm @ dm.T
rel, irr = [], []
for i, q in enumerate(qids):
    for j in np.argsort(-S[i])[:K]:
        (rel if eval_docs[j] in qr[q] else irr).append(S[i, j])
rel, irr = np.array(rel), np.array(irr)
y = np.r_[np.ones(len(rel)), np.zeros(len(irr))]; s = np.r_[rel, irr]
auc = roc_auc_score(y, s)
print("rel", len(rel), "irr", len(irr), "auc", auc, "min_rel", rel.min(), "max_irr", irr.max())
bins = np.linspace(min(s.min(), 0), max(s.max(), 1), 80)
fig, ax = plt.subplots(figsize=(9, 5.4))
ax.hist(irr, bins=bins, density=True, alpha=0.5, label="Non-relevant")
ax.hist(rel, bins=bins, density=True, alpha=0.5, label="Relevant")
ax.set_title(f"SciFact Fine-tuned Embedding (MiniLM)\nRelevant vs Non-relevant  (ROC-AUC={auc:.3f})")
ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density"); ax.legend()
fig.tight_layout(); fig.savefig(OUT + "/ft_embedding_distribution.png", dpi=150)
