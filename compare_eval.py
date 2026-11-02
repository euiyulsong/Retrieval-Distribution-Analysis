# All methods scored on the SAME eval set:
#   SciFact TEST queries (with qrels), candidates = BM25 top-100 per query,
#   label = test qrel. Cross-encoder FT is trained on train split.
import os, re, json, random
import numpy as np, ir_datasets, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

SEED = 42; random.seed(SEED); np.random.seed(SEED)
OUT = "/tmp/claude-1000/-home-euiyul/b2cd91ee-ceb0-4d80-8cdc-476fc73249d3/scratchpad/emb"
TOP_K = 100
CE_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_FT_DIR = OUT + "/ce_ft_model"
DENSE_BASE = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_FT_DIR = OUT + "/ft_model"   # from embed_relevance.py

tr = ir_datasets.load("beir/scifact/train"); te = ir_datasets.load("beir/scifact/test")
docs = {}
for d in tr.docs_iter():
    docs[str(d.doc_id)] = (((getattr(d,"title","") or "")+" "+(getattr(d,"text","") or "")).strip())
doc_ids = list(docs)
def qrels(ds):
    m = defaultdict(set)
    for r in ds.qrels_iter():
        if r.relevance > 0: m[str(r.query_id)].add(str(r.doc_id))
    return m
tr_q = {str(q.query_id): q.text for q in tr.queries_iter()}
te_q = {str(q.query_id): q.text for q in te.queries_iter()}
tr_qr, te_qr = qrels(tr), qrels(te)
te_qids = [q for q in te_qr if q in te_q]
print("test queries:", len(te_qids))

def tok(t): return re.findall(r"\w+", t.lower())
bm25 = BM25Okapi([tok(docs[d]) for d in doc_ids])

# ---- eval candidates: BM25 top-100 per test query ----
cands = {}
for q in te_qids:
    s = np.asarray(bm25.get_scores(tok(te_q[q])))
    cands[q] = [int(i) for i in np.argsort(-s)[:TOP_K]]
eval_pairs = [(q, i) for q in te_qids for i in cands[q]]
y = np.array([int(doc_ids[i] in te_qr[q]) for q, i in eval_pairs])
print("eval pairs:", len(eval_pairs), "relevant:", int(y.sum()))
bm25_s = np.array([bm25.get_scores(tok(te_q[q]))[i] for q, i in eval_pairs])  # cheap enough

def scores_dense(path):
    m = SentenceTransformer(path)
    qs = sorted(set(q for q, _ in eval_pairs))
    qe = dict(zip(qs, m.encode([te_q[q] for q in qs], normalize_embeddings=True, batch_size=128, show_progress_bar=False)))
    need = sorted(set(i for _, i in eval_pairs))
    de = dict(zip(need, m.encode([docs[doc_ids[i]] for i in need], normalize_embeddings=True, batch_size=128, show_progress_bar=False)))
    return np.array([float(qe[q] @ de[i]) for q, i in eval_pairs])

def scores_ce(path):
    m = CrossEncoder(path)
    return np.asarray(m.predict([(te_q[q], docs[doc_ids[i]]) for q, i in eval_pairs], batch_size=128, show_progress_bar=False))

dense_pre = scores_dense(DENSE_BASE)
dense_ft = scores_dense(DENSE_FT_DIR)
ce_pre = scores_ce(CE_BASE)

# ---- train cross-encoder FT on train split (BCE, 4 random BM25-hard negs per positive) ----
if not os.path.exists(CE_FT_DIR):
    rows = []
    for q, rel in tr_qr.items():
        if q not in tr_q: continue
        s = np.asarray(bm25.get_scores(tok(tr_q[q])))
        top = [doc_ids[i] for i in np.argsort(-s)[:TOP_K] if doc_ids[i] not in rel]
        negs = random.Random(SEED).sample(top, min(4 * len(rel), len(top)))
        for d in rel:
            if d in docs:
                rows.append({"sentence1": tr_q[q], "sentence2": docs[d], "label": 1.0})
        for d in negs:
            rows.append({"sentence1": tr_q[q], "sentence2": docs[d], "label": 0.0})
    print("CE train rows:", len(rows), "pos:", sum(r["label"] for r in rows))
    ce = CrossEncoder(CE_BASE, num_labels=1)
    args = CrossEncoderTrainingArguments(output_dir=OUT + "/ce_ckpt", num_train_epochs=3,
        per_device_train_batch_size=32, learning_rate=2e-5, warmup_ratio=0.1, seed=SEED,
        report_to="none", save_strategy="no")
    CrossEncoderTrainer(model=ce, args=args, train_dataset=Dataset.from_list(rows),
                        loss=BinaryCrossEntropyLoss(ce)).train()
    ce.save(CE_FT_DIR)
ce_ft = scores_ce(CE_FT_DIR)

methods = [("BM25", bm25_s), ("Dense (pretrained)", dense_pre), ("Dense (fine-tuned)", dense_ft),
           ("Cross-Encoder (pretrained)", ce_pre), ("Cross-Encoder (fine-tuned)", ce_ft)]
res = {}
for name, s in methods:
    auc = roc_auc_score(y, s)
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    k = int(np.argmax(f1[:-1]))
    res[name] = {"roc_auc": float(auc), "best_f1": float(f1[k]), "best_thr": float(t[k]),
                 "mean_rel": float(s[y==1].mean()), "mean_irrel": float(s[y==0].mean()),
                 "min_rel": float(s[y==1].min()), "max_irrel": float(s[y==0].max())}
    print(name, json.dumps(res[name]))
json.dump(res, open(OUT + "/compare_eval_results.json", "w"), indent=2)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes.flat[-1].axis("off")
for ax, (name, s) in zip(axes.flat, methods):
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 70)
    ax.hist(s[y==0], bins=bins, density=True, alpha=0.5, label="Non-relevant")
    ax.hist(s[y==1], bins=bins, density=True, alpha=0.5, label="Relevant")
    ax.set_title(f"{name}\nROC-AUC={res[name]['roc_auc']:.3f} · best F1={res[name]['best_f1']:.3f}", fontsize=12)
    ax.set_xlabel("score"); ax.set_ylabel("Density"); ax.legend()
fig.suptitle("SciFact TEST, same candidates (BM25 top-100): relevant vs non-relevant by method", fontsize=14)
fig.tight_layout(); fig.savefig(OUT + "/method_comparison_same_eval.png", dpi=120)
