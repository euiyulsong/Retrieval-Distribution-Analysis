# ============================================================
# SciFact:
# BM25 vs Dense Pretrained vs Dense Fine-tuned
#
# Train:
#   beir/scifact/train
#
# Test:
#   beir/scifact/test
#
# Outputs:
#   - score distribution plots
#   - threshold sweep plots
#   - top1 score distribution
#   - test retrieval metrics
#   - extreme low-score TP / high-score FP examples
#   - CSV files
#
# Install:
#
# pip install -U \
#   ir_datasets \
#   sentence-transformers \
#   datasets \
#   rank-bm25 \
#   matplotlib \
#   pandas \
#   numpy \
#   scikit-learn \
#   tqdm \
#   accelerate
#
# ============================================================


import os
import re
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

import ir_datasets

from rank_bm25 import BM25Okapi

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from sentence_transformers.training_args import BatchSamplers

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OUTPUT_DIR = "./scifact_threshold_experiment"

FINETUNED_MODEL_DIR = (
    OUTPUT_DIR + "/finetuned_model"
)

TOP_K = 100

TRAIN_EPOCHS = 3

TRAIN_BATCH_SIZE = 64

LEARNING_RATE = 2e-5

ENCODE_BATCH_SIZE = 128

# False면 이미 저장된 FT model 재사용
FORCE_TRAIN = True


# ============================================================
# SEED
# ============================================================

random.seed(SEED)

np.random.seed(SEED)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD SCIFACT
# ============================================================

print()
print("=" * 100)
print("LOAD SCIFACT")
print("=" * 100)


train_dataset_irds = ir_datasets.load(
    "beir/scifact/train"
)

test_dataset_irds = ir_datasets.load(
    "beir/scifact/test"
)


# ------------------------------------------------------------
# documents
#
# train/test corpus는 동일한 SciFact corpus를 사용.
# ------------------------------------------------------------

docs = {}

for doc in train_dataset_irds.docs_iter():

    title = (
        getattr(
            doc,
            "title",
            ""
        )
        or ""
    )

    text = (
        getattr(
            doc,
            "text",
            ""
        )
        or ""
    )

    docs[str(doc.doc_id)] = (
        title
        + " "
        + text
    ).strip()


doc_ids = list(
    docs.keys()
)


print(
    "docs:",
    len(docs)
)


# ============================================================
# LOAD TRAIN QUERIES / QRELS
# ============================================================

train_queries = {
    str(q.query_id): q.text
    for q
    in train_dataset_irds.queries_iter()
}


train_qrels = defaultdict(set)

for qr in train_dataset_irds.qrels_iter():

    if qr.relevance > 0:

        train_qrels[
            str(qr.query_id)
        ].add(
            str(qr.doc_id)
        )


print(
    "train queries:",
    len(train_queries)
)

print(
    "train qrel queries:",
    len(train_qrels)
)


# ============================================================
# LOAD TEST QUERIES / QRELS
# ============================================================

test_queries = {
    str(q.query_id): q.text
    for q
    in test_dataset_irds.queries_iter()
}


test_qrels = defaultdict(set)

for qr in test_dataset_irds.qrels_iter():

    if qr.relevance > 0:

        test_qrels[
            str(qr.query_id)
        ].add(
            str(qr.doc_id)
        )


print(
    "test queries:",
    len(test_queries)
)

print(
    "test qrel queries:",
    len(test_qrels)
)


# ============================================================
# SANITY: TRAIN / TEST QUERY OVERLAP
# ============================================================

overlap = (
    set(train_queries)
    &
    set(test_queries)
)

print(
    "train/test query overlap:",
    len(overlap)
)


# ============================================================
# BUILD TRAIN PAIRS
#
# MultipleNegativesRankingLoss:
#
# anchor   = query
# positive = relevant document
#
# qrel이 여러 개면 모두 training pair로 사용.
# ============================================================

train_anchors = []
train_positives = []

for qid, relevant_docs in train_qrels.items():

    if qid not in train_queries:
        continue

    query = train_queries[
        qid
    ]

    for did in relevant_docs:

        if did not in docs:
            continue

        train_anchors.append(
            query
        )

        train_positives.append(
            docs[did]
        )


print()
print("=" * 100)
print("TRAIN PAIRS")
print("=" * 100)

print(
    "pairs:",
    len(train_anchors)
)


print()
print("Examples:")

for i in range(
    min(
        3,
        len(train_anchors)
    )
):

    print()

    print(
        "QUERY:",
        train_anchors[i]
    )

    print(
        "POSITIVE:",
        train_positives[i][:300]
    )


# ============================================================
# HF Dataset
# ============================================================

st_train_dataset = Dataset.from_dict({
    "anchor": train_anchors,
    "positive": train_positives,
})


# ============================================================
# TRAIN DENSE MODEL
# ============================================================

print()
print("=" * 100)
print("TRAIN DENSE RETRIEVER")
print("=" * 100)


if (
    FORCE_TRAIN
    or
    not os.path.exists(
        FINETUNED_MODEL_DIR
    )
):

    ft_model = SentenceTransformer(
        BASE_MODEL
    )

    train_loss = (
        losses.MultipleNegativesRankingLoss(
            ft_model
        )
    )

    args = SentenceTransformerTrainingArguments(

        output_dir=(
            OUTPUT_DIR
            + "/checkpoints"
        ),

        num_train_epochs=(
            TRAIN_EPOCHS
        ),

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        learning_rate=(
            LEARNING_RATE
        ),

        warmup_ratio=0.1,

        fp16=True,

        bf16=False,

        batch_sampler=(
            BatchSamplers.NO_DUPLICATES
        ),

        save_strategy="epoch",

        logging_steps=10,

        report_to="none",

        seed=SEED,
    )


    trainer = SentenceTransformerTrainer(

        model=ft_model,

        args=args,

        train_dataset=(
            st_train_dataset
        ),

        loss=train_loss,
    )


    trainer.train()


    ft_model.save(
        FINETUNED_MODEL_DIR
    )


else:

    print(
        "Using existing model:",
        FINETUNED_MODEL_DIR
    )


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print()
print("=" * 100)
print("LOAD MODELS FOR EVAL")
print("=" * 100)


pretrained_model = SentenceTransformer(
    BASE_MODEL
)

finetuned_model = SentenceTransformer(
    FINETUNED_MODEL_DIR
)


# ============================================================
# BM25
# ============================================================

def tokenize(text):

    return re.findall(
        r"\w+",
        text.lower()
    )


print()
print(
    "Building BM25..."
)

tokenized_docs = [
    tokenize(
        docs[did]
    )
    for did
    in doc_ids
]


bm25 = BM25Okapi(
    tokenized_docs
)


# ============================================================
# ENCODE DOCUMENTS
# ============================================================

doc_texts = [
    docs[d]
    for d
    in doc_ids
]


print()
print(
    "Encoding PRETRAINED documents..."
)

pretrained_doc_emb = (
    pretrained_model.encode(

        doc_texts,

        batch_size=(
            ENCODE_BATCH_SIZE
        ),

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )
)


print()
print(
    "Encoding FINETUNED documents..."
)

finetuned_doc_emb = (
    finetuned_model.encode(

        doc_texts,

        batch_size=(
            ENCODE_BATCH_SIZE
        ),

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )
)


# ============================================================
# EVALUATE
# ============================================================

rows = []

retrieval_metrics = defaultdict(
    lambda: defaultdict(float)
)


def add_candidates(
    retriever_name,
    qid,
    query,
    scores,
    relevant,
):

    order = np.argsort(
        scores
    )[::-1]

    top_idx = order[
        :TOP_K
    ]

    for rank, i in enumerate(
        top_idx,
        start=1
    ):

        did = doc_ids[i]

        rows.append({

            "qid": qid,

            "query": query,

            "doc_id": did,

            "retriever": (
                retriever_name
            ),

            "rank": rank,

            "score": float(
                scores[i]
            ),

            "label": int(
                did
                in relevant
            ),
        })


def evaluate_ranking(
    order,
    relevant,
):

    ranked_ids = [
        doc_ids[i]
        for i
        in order
    ]

    hits = [
        int(
            did
            in relevant
        )
        for did
        in ranked_ids
    ]


    # MRR
    rr = 0.0

    for rank, hit in enumerate(
        hits,
        start=1
    ):

        if hit:

            rr = (
                1.0
                /
                rank
            )

            break


    # Recall
    def recall_at(k):

        found = len(
            set(
                ranked_ids[:k]
            )
            &
            relevant
        )

        return (
            found
            /
            len(relevant)
        )


    return {

        "mrr": rr,

        "recall@1": (
            recall_at(1)
        ),

        "recall@5": (
            recall_at(5)
        ),

        "recall@10": (
            recall_at(10)
        ),

        "recall@100": (
            recall_at(100)
        ),
    }


print()
print("=" * 100)
print("TEST RETRIEVAL")
print("=" * 100)


eval_qids = list(
    test_qrels.keys()
)


metric_rows = []


for qid in tqdm(
    eval_qids
):

    query = test_queries[
        qid
    ]

    relevant = test_qrels[
        qid
    ]


    # ========================================================
    # BM25
    # ========================================================

    bm25_scores = np.asarray(
        bm25.get_scores(
            tokenize(query)
        ),
        dtype=np.float32,
    )


    bm25_order = np.argsort(
        bm25_scores
    )[::-1]


    add_candidates(

        "BM25",

        qid,

        query,

        bm25_scores,

        relevant,
    )


    metrics = evaluate_ranking(
        bm25_order,
        relevant
    )

    metric_rows.append({

        "qid": qid,

        "retriever": "BM25",

        **metrics,
    })


    # ========================================================
    # PRETRAINED
    # ========================================================

    q_pre = pretrained_model.encode(

        query,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )


    pre_scores = (
        pretrained_doc_emb
        @
        q_pre
    )


    pre_order = np.argsort(
        pre_scores
    )[::-1]


    add_candidates(

        "Dense_Pretrained",

        qid,

        query,

        pre_scores,

        relevant,
    )


    metrics = evaluate_ranking(
        pre_order,
        relevant
    )

    metric_rows.append({

        "qid": qid,

        "retriever": (
            "Dense_Pretrained"
        ),

        **metrics,
    })


    # ========================================================
    # FINETUNED
    # ========================================================

    q_ft = finetuned_model.encode(

        query,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )


    ft_scores = (
        finetuned_doc_emb
        @
        q_ft
    )


    ft_order = np.argsort(
        ft_scores
    )[::-1]


    add_candidates(

        "Dense_Finetuned",

        qid,

        query,

        ft_scores,

        relevant,
    )


    metrics = evaluate_ranking(
        ft_order,
        relevant
    )

    metric_rows.append({

        "qid": qid,

        "retriever": (
            "Dense_Finetuned"
        ),

        **metrics,
    })


# ============================================================
# SAVE SCORES
# ============================================================

df = pd.DataFrame(
    rows
)


scores_path = (
    OUTPUT_DIR
    +
    "/scifact_scores.csv"
)


df.to_csv(
    scores_path,
    index=False
)


# ============================================================
# RANKING METRICS
# ============================================================

metric_df = pd.DataFrame(
    metric_rows
)


metric_summary = (
    metric_df
    .groupby(
        "retriever"
    )[
        [
            "mrr",
            "recall@1",
            "recall@5",
            "recall@10",
            "recall@100",
        ]
    ]
    .mean()
)


print()
print("=" * 100)
print("RETRIEVAL METRICS")
print("=" * 100)

print(
    metric_summary
)


metric_summary.to_csv(
    OUTPUT_DIR
    +
    "/retrieval_metrics.csv"
)


# ============================================================
# SCORE STATISTICS
# ============================================================

print()
print("=" * 100)
print("SCORE DISTRIBUTIONS")
print("=" * 100)


stats = (
    df.groupby(
        [
            "retriever",
            "label"
        ]
    )[
        "score"
    ]
    .describe()
)


print(
    stats
)


stats.to_csv(
    OUTPUT_DIR
    +
    "/score_statistics.csv"
)


# ============================================================
# HISTOGRAM:
# Relevant vs Non-Relevant
# ============================================================

retrievers = [

    "BM25",

    "Dense_Pretrained",

    "Dense_Finetuned",
]


for retriever in retrievers:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    positive = (
        tmp[
            tmp.label
            ==
            1
        ]
        .score
        .to_numpy()
    )


    negative = (
        tmp[
            tmp.label
            ==
            0
        ]
        .score
        .to_numpy()
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(

        negative,

        bins=80,

        alpha=0.5,

        density=True,

        label="Non-relevant",
    )


    plt.hist(

        positive,

        bins=80,

        alpha=0.5,

        density=True,

        label="Relevant",
    )


    plt.xlabel(
        "Retrieval score"
    )

    plt.ylabel(
        "Density"
    )


    plt.title(
        f"SciFact {retriever}\n"
        "Relevant vs Non-relevant"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        +
        f"/{retriever.lower()}_distribution.png",

        dpi=180,
    )


    plt.close()


# ============================================================
# THRESHOLD SWEEP
# ============================================================

def threshold_sweep(
    scores,
    labels,
    n=500,
):

    min_score = float(
        np.min(scores)
    )

    max_score = float(
        np.max(scores)
    )


    thresholds = np.linspace(

        min_score,

        max_score,

        n,
    )


    output = []


    for t in thresholds:

        pred = (
            scores
            >=
            t
        ).astype(int)


        tp = int(
            np.sum(
                (pred == 1)
                &
                (labels == 1)
            )
        )


        fp = int(
            np.sum(
                (pred == 1)
                &
                (labels == 0)
            )
        )


        tn = int(
            np.sum(
                (pred == 0)
                &
                (labels == 0)
            )
        )


        fn = int(
            np.sum(
                (pred == 0)
                &
                (labels == 1)
            )
        )


        precision = (

            tp
            /
            (tp + fp)

            if (
                tp + fp
            )

            else 1.0
        )


        recall = (

            tp
            /
            (tp + fn)

            if (
                tp + fn
            )

            else 0.0
        )


        specificity = (

            tn
            /
            (tn + fp)

            if (
                tn + fp
            )

            else 0.0
        )


        f1 = (

            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )

            if (
                precision
                +
                recall
            )

            else 0.0
        )


        output.append({

            "threshold": (
                float(t)
            ),

            "precision": (
                precision
            ),

            "recall": (
                recall
            ),

            "specificity": (
                specificity
            ),

            "f1": f1,

            "tp": tp,

            "fp": fp,

            "tn": tn,

            "fn": fn,
        })


    return pd.DataFrame(
        output
    )


threshold_summary = []


for retriever in retrievers:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    scores = (
        tmp.score
        .to_numpy(
            dtype=float
        )
    )


    labels = (
        tmp.label
        .to_numpy(
            dtype=int
        )
    )


    sweep = threshold_sweep(
        scores,
        labels
    )


    sweep.to_csv(

        OUTPUT_DIR
        +
        f"/{retriever.lower()}_thresholds.csv",

        index=False,
    )


    best = sweep.loc[
        sweep.f1.idxmax()
    ]


    auc = roc_auc_score(
        labels,
        scores
    )


    threshold_summary.append({

        "retriever": (
            retriever
        ),

        "best_threshold": (
            best["threshold"]
        ),

        "precision": (
            best["precision"]
        ),

        "recall": (
            best["recall"]
        ),

        "f1": (
            best["f1"]
        ),

        "roc_auc": auc,
    })


    print()
    print("=" * 100)

    print(
        retriever
    )

    print("=" * 100)

    print(
        "Best threshold :",
        best["threshold"]
    )

    print(
        "Precision      :",
        best["precision"]
    )

    print(
        "Recall         :",
        best["recall"]
    )

    print(
        "F1             :",
        best["f1"]
    )

    print(
        "ROC AUC        :",
        auc
    )


    # --------------------------------------------------------
    # matplotlib/pandas compatibility fix:
    # 반드시 numpy conversion
    # --------------------------------------------------------

    x = (
        sweep[
            "threshold"
        ]
        .to_numpy()
    )


    p = (
        sweep[
            "precision"
        ]
        .to_numpy()
    )


    r = (
        sweep[
            "recall"
        ]
        .to_numpy()
    )


    f = (
        sweep[
            "f1"
        ]
        .to_numpy()
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        x,
        p,
        label="Precision"
    )


    plt.plot(
        x,
        r,
        label="Recall"
    )


    plt.plot(
        x,
        f,
        label="F1"
    )


    plt.axvline(

        float(
            best[
                "threshold"
            ]
        ),

        linestyle="--",

        label="Best F1 threshold",
    )


    plt.xlabel(
        "Absolute score threshold"
    )


    plt.ylabel(
        "Metric"
    )


    plt.title(
        f"SciFact {retriever}\n"
        "Global Threshold Sweep"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        +
        f"/{retriever.lower()}_threshold_curve.png",

        dpi=180,
    )


    plt.close()


# ============================================================
# SAVE THRESHOLD SUMMARY
# ============================================================

threshold_summary_df = pd.DataFrame(
    threshold_summary
)


print()
print("=" * 100)
print("THRESHOLD SUMMARY")
print("=" * 100)

print(
    threshold_summary_df
    .to_string(
        index=False
    )
)


threshold_summary_df.to_csv(

    OUTPUT_DIR
    +
    "/threshold_summary.csv",

    index=False,
)


# ============================================================
# TOP1 SCORE DISTRIBUTION
# ============================================================

for retriever in retrievers:

    top1 = df[
        (
            df.retriever
            ==
            retriever
        )
        &
        (
            df["rank"]
            ==
            1
        )
    ]


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(

        top1.score.to_numpy(),

        bins=50,
    )


    plt.xlabel(
        "Top-1 score"
    )


    plt.ylabel(
        "Number of queries"
    )


    plt.title(
        f"SciFact {retriever}\n"
        "Query-wise Top-1 Score Distribution"
    )


    plt.tight_layout()


    plt.savefig(

        OUTPUT_DIR
        +
        f"/{retriever.lower()}_top1.png",

        dpi=180,
    )


    plt.close()


# ============================================================
# EXTREME COUNTEREXAMPLES
# ============================================================

print()
print("=" * 100)
print("EXTREME COUNTEREXAMPLES")
print("=" * 100)


counterexample_rows = []


for retriever in retrievers:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    positives = (

        tmp[
            tmp.label
            ==
            1
        ]

        .sort_values(
            "score"
        )

        .head(15)
    )


    negatives = (

        tmp[
            tmp.label
            ==
            0
        ]

        .sort_values(
            "score",
            ascending=False
        )

        .head(15)
    )


    print()
    print("#" * 100)

    print(
        retriever
    )

    print("#" * 100)


    print()
    print(
        "LOW-SCORE TRUE POSITIVES"
    )


    for _, row in positives.iterrows():

        print()
        print(
            "score:",
            round(
                float(
                    row["score"]
                ),
                4
            )
        )

        print(
            "rank:",
            row["rank"]
        )

        print(
            "query:",
            row["query"]
        )

        print(
            "doc:",
            docs[
                row["doc_id"]
            ][:400]
        )


        counterexample_rows.append({

            "retriever": retriever,

            "type": (
                "low_score_true_positive"
            ),

            "score": (
                row["score"]
            ),

            "rank": (
                row["rank"]
            ),

            "query": (
                row["query"]
            ),

            "doc_id": (
                row["doc_id"]
            ),

            "document": (
                docs[
                    row["doc_id"]
                ]
            ),
        })


    print()
    print(
        "HIGH-SCORE FALSE POSITIVES"
    )


    for _, row in negatives.iterrows():

        print()
        print(
            "score:",
            round(
                float(
                    row["score"]
                ),
                4
            )
        )

        print(
            "rank:",
            row["rank"]
        )

        print(
            "query:",
            row["query"]
        )

        print(
            "doc:",
            docs[
                row["doc_id"]
            ][:400]
        )


        counterexample_rows.append({

            "retriever": retriever,

            "type": (
                "high_score_false_positive"
            ),

            "score": (
                row["score"]
            ),

            "rank": (
                row["rank"]
            ),

            "query": (
                row["query"]
            ),

            "doc_id": (
                row["doc_id"]
            ),

            "document": (
                docs[
                    row["doc_id"]
                ]
            ),
        })


pd.DataFrame(
    counterexample_rows
).to_csv(

    OUTPUT_DIR
    +
    "/counterexamples.csv",

    index=False,
)


# ============================================================
# SCORE OVERLAP ANALYSIS
#
# 여기서 진짜 핵심:
#
# minimum positive
# maximum negative
#
# max_negative > min_positive면
# absolute threshold 하나로 perfect separation 불가능.
# ============================================================

print()
print("=" * 100)
print("SCORE OVERLAP")
print("=" * 100)


overlap_rows = []


for retriever in retrievers:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    pos = (
        tmp[
            tmp.label
            ==
            1
        ]
        .score
        .to_numpy()
    )


    neg = (
        tmp[
            tmp.label
            ==
            0
        ]
        .score
        .to_numpy()
    )


    min_positive = float(
        pos.min()
    )


    max_negative = float(
        neg.max()
    )


    overlap_amount = (
        max_negative
        -
        min_positive
    )


    perfectly_separable = (
        max_negative
        <
        min_positive
    )


    print()

    print(
        retriever
    )

    print(
        "min positive:",
        min_positive
    )

    print(
        "max negative:",
        max_negative
    )

    print(
        "overlap amount:",
        overlap_amount
    )

    print(
        "perfect global threshold possible:",
        perfectly_separable
    )


    overlap_rows.append({

        "retriever": (
            retriever
        ),

        "min_positive": (
            min_positive
        ),

        "max_negative": (
            max_negative
        ),

        "overlap_amount": (
            overlap_amount
        ),

        "perfectly_separable": (
            perfectly_separable
        ),
    })


pd.DataFrame(
    overlap_rows
).to_csv(

    OUTPUT_DIR
    +
    "/score_overlap.csv",

    index=False,
)


# ============================================================
# OPTIONAL:
# DENSE-specific fixed cosine thresholds
#
# 우리가 지금 대화했던
# 0.1 / 0.2 / 0.3 / 0.4 / ...
# 를 실제 benchmark에서 확인.
# ============================================================

print()
print("=" * 100)
print("FIXED COSINE THRESHOLDS")
print("=" * 100)


fixed_thresholds = [

    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
]


fixed_rows = []


for retriever in [

    "Dense_Pretrained",

    "Dense_Finetuned",
]:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    scores = (
        tmp.score
        .to_numpy()
    )


    labels = (
        tmp.label
        .to_numpy()
    )


    for threshold in fixed_thresholds:

        pred = (
            scores
            >=
            threshold
        )


        tp = np.sum(
            pred
            &
            (
                labels
                ==
                1
            )
        )


        fp = np.sum(
            pred
            &
            (
                labels
                ==
                0
            )
        )


        fn = np.sum(
            (~pred)
            &
            (
                labels
                ==
                1
            )
        )


        precision = (

            tp
            /
            (
                tp
                +
                fp
            )

            if (
                tp + fp
            )

            else 1.0
        )


        recall = (

            tp
            /
            (
                tp
                +
                fn
            )

            if (
                tp + fn
            )

            else 0.0
        )


        f1 = (

            2
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )

            if (
                precision
                +
                recall
            )

            else 0.0
        )


        fixed_rows.append({

            "retriever": (
                retriever
            ),

            "threshold": (
                threshold
            ),

            "precision": (
                precision
            ),

            "recall": (
                recall
            ),

            "f1": (
                f1
            ),

            "tp": int(tp),

            "fp": int(fp),

            "fn": int(fn),
        })


fixed_df = pd.DataFrame(
    fixed_rows
)


print(
    fixed_df.to_string(
        index=False
    )
)


fixed_df.to_csv(

    OUTPUT_DIR
    +
    "/fixed_cosine_thresholds.csv",

    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)


print()
print(
    "[Retrieval quality]"
)

print(
    metric_summary
)


print()
print(
    "[Threshold classification]"
)

print(
    threshold_summary_df
    .to_string(
        index=False
    )
)


print()
print(
    "[Score overlap]"
)

print(
    pd.DataFrame(
        overlap_rows
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Saved under:",
    OUTPUT_DIR
)

print("""
Files:

  finetuned_model/

  scifact_scores.csv

  retrieval_metrics.csv
  score_statistics.csv
  threshold_summary.csv
  score_overlap.csv
  fixed_cosine_thresholds.csv
  counterexamples.csv

  bm25_distribution.png
  dense_pretrained_distribution.png
  dense_finetuned_distribution.png

  bm25_threshold_curve.png
  dense_pretrained_threshold_curve.png
  dense_finetuned_threshold_curve.png

  bm25_top1.png
  dense_pretrained_top1.png
  dense_finetuned_top1.png
""")
