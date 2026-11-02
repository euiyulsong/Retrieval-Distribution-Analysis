# SciFact: relevant vs irrelevant embedding space, before / after contrastive fine-tuning
import os, random, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ir_datasets
from collections import defaultdict
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
BASE = "sentence-transformers/all-MiniLM-L6-v2"
OUT = "/tmp/claude-1000/-home-euiyul/b2cd91ee-ceb0-4d80-8cdc-476fc73249d3/scratchpad/emb"
FT_DIR = OUT + "/ft_model"
N_HARD = 3          # hard negatives per (query, positive) pair
MAX_EVAL_Q = 150    # test queries used for the picture
K_CAND = 30         # candidates per test query (non-relevant ones = "irrelevant")

tr = ir_datasets.load("beir/scifact/train")
te = ir_datasets.load("beir/scifact/test")
docs = {str(d.doc_id): (((getattr(d, "title", "") or "") + " " + (getattr(d, "text", "") or "")).strip()) for d in tr.docs_iter()}
doc_ids = list(docs)
tr_q = {str(q.query_id): q.text for q in tr.queries_iter()}
te_q = {str(q.query_id): q.text for q in te.queries_iter()}
def qrels(ds):
    m = defaultdict(set)
    for r in ds.qrels_iter():
        if r.relevance > 0: m[str(r.query_id)].add(str(r.doc_id))
    return m
tr_qr, te_qr = qrels(tr), qrels(te)

# ---- training triplets: (query, relevant doc, hard irrelevant doc) ----
base = SentenceTransformer(BASE)
tr_qids = [q for q in tr_qr if q in tr_q]
tr_doc_mat = base.encode([docs[d] for d in doc_ids], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
tr_q_mat = base.encode([tr_q[q] for q in tr_qids], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
A, P, N = [], [], []
for i, q in enumerate(tr_qids):
    sims = tr_doc_mat @ tr_q_mat[i]
    order = np.argsort(-sims)
    hard = [doc_ids[j] for j in order[:50] if doc_ids[j] not in tr_qr[q]][:N_HARD]
    for pos in tr_qr[q]:
        if pos not in docs: continue
        for neg in hard:
            A.append(tr_q[q]); P.append(docs[pos]); N.append(docs[neg])
print("triplets:", len(A))
ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": N}).shuffle(seed=SEED)

if not os.path.exists(FT_DIR):
    model = SentenceTransformer(BASE)
    loss = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=OUT + "/ckpt", num_train_epochs=2, per_device_train_batch_size=64,
        learning_rate=2e-5, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=SEED, report_to="none", save_strategy="no", logging_steps=50)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save(FT_DIR)
ft = SentenceTransformer(FT_DIR)

# ---- evaluation set: test queries; candidates = top-K from base dense; label = qrel ----
te_ids = [q for q in te_qr if q in te_q][:MAX_EVAL_Q]
test_qids_all = list(te_qr.keys())
need = set()
for q in te_ids:
    need |= te_qr[q]
extra = random.Random(SEED).sample(doc_ids, 300)
eval_docs = sorted(need | set(extra))

def encode_all(m):
    qm = m.encode([te_q[q] for q in te_ids], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    dm = m.encode([docs[d] for d in eval_docs], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    return qm, dm

def cand_pairs(qm, dm):
    # for each query: top-K docs of this model -> label relevant/irrelevant
    rows = []
    S = qm @ dm.T
    for i, q in enumerate(te_ids):
        for j in np.argsort(-S[i])[:K_CAND]:
            rows.append((i, j, eval_docs[j] in te_qr[q], S[i, j]))
    return rows

res = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, (name, m) in zip(axes, [("Pretrained", base), ("Fine-tuned (relevant vs irrelevant)", ft)]):
    qm, dm = encode_all(m)
    rows = cand_pairs(qm, dm)
    y = np.array([r[2] for r in rows], dtype=int)
    s = np.array([r[3] for r in rows])
    auc = roc_auc_score(y, s)
    # 2D projection of the same points: test queries + candidate docs (rel / irrel)
    cand_idx = sorted({r[1] for r in rows})
    X = np.vstack([qm, dm[cand_idx]])
    Z = TSNE(n_components=2, random_state=SEED, metric="cosine", init="pca", perplexity=30).fit_transform(X)
    zq, zd = Z[:len(qm)], Z[len(qm):]
    pos_map = {j: k for k, j in enumerate(cand_idx)}
    is_rel = np.zeros(len(cand_idx), dtype=bool)
    for (i, j, lab, _) in rows:
        if lab: is_rel[pos_map[j]] = True
    ax.scatter(zd[~is_rel, 0], zd[~is_rel, 1], s=10, c="#9aa0a6", alpha=0.45, label="irrelevant doc")
    ax.scatter(zd[is_rel, 0], zd[is_rel, 1], s=14, c="#1a73e8", alpha=0.8, label="relevant doc")
    ax.scatter(zq[:, 0], zq[:, 1], s=18, c="#d93025", marker="^", alpha=0.8, label="query")
    ax.set_title(f"{name}\nROC-AUC (cos, top-{K_CAND} candidates) = {auc:.3f}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    res[name] = {"roc_auc": float(auc),
                 "mean_cos_rel": float(s[y == 1].mean()) if (y == 1).any() else None,
                 "mean_cos_irrel": float(s[y == 0].mean()),
                 "n_rel": int(y.sum()), "n_irrel": int((y == 0).sum())}
fig.suptitle("SciFact: query / document embeddings (t-SNE) before vs after relevance fine-tuning", fontsize=13)
fig.tight_layout()
fig.savefig(OUT + "/embedding_relevance_tsne.png", dpi=150)
print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open(OUT + "/results.json", "w"), indent=2, ensure_ascii=False)
