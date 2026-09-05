# ============================================================
# SciFact Cross-Encoder:
# Pretrained vs Fine-tuned
#
# Train:
#   beir/scifact/train
#
# Test:
#   beir/scifact/test
#
# Candidate generation:
#   BM25 top-K
#
# Cross-encoder:
#   cross-encoder/ms-marco-MiniLM-L-6-v2
#
# Outputs:
#   - crossencoder_pretrained_distribution.png
#   - crossencoder_finetuned_distribution.png
#   - crossencoder_pretrained_threshold_curve.png
#   - crossencoder_finetuned_threshold_curve.png
#   - crossencoder_pretrained_top1.png
#   - crossencoder_finetuned_top1.png
#   - threshold_summary.csv
#   - score_overlap.csv
#   - counterexamples.csv
#
# install:
#
# pip install -U \
#   ir_datasets \
#   sentence-transformers \
#   rank-bm25 \
#   matplotlib \
#   pandas \
#   numpy \
#   scikit-learn \
#   tqdm \
#   torch
# ============================================================


import os
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import ir_datasets

from rank_bm25 import BM25Okapi
from sklearn.metrics import roc_auc_score

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
)
from datasets import Dataset


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

OUTPUT_DIR = "./scifact_crossencoder_experiment"

FINETUNED_MODEL_DIR = (
    OUTPUT_DIR + "/finetuned_model"
)

TOP_K = 100

# train negative sampling
NEG_PER_POSITIVE = 4

TRAIN_EPOCHS = 3

TRAIN_BATCH_SIZE = 32

EVAL_BATCH_SIZE = 128

LEARNING_RATE = 2e-5

FORCE_TRAIN = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

train_irds = ir_datasets.load(
    "beir/scifact/train"
)

test_irds = ir_datasets.load(
    "beir/scifact/test"
)


# ============================================================
# DOCS
# ============================================================

docs = {}

for doc in train_irds.docs_iter():

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

doc_texts = [
    docs[d]
    for d in doc_ids
]

print(
    "docs:",
    len(docs)
)


# ============================================================
# TRAIN QUERIES / QRELS
# ============================================================

train_queries = {
    str(q.query_id): q.text
    for q
    in train_irds.queries_iter()
}

train_qrels = defaultdict(set)

for qr in train_irds.qrels_iter():

    if qr.relevance > 0:

        train_qrels[
            str(qr.query_id)
        ].add(
            str(qr.doc_id)
        )


# ============================================================
# TEST QUERIES / QRELS
# ============================================================

test_queries = {
    str(q.query_id): q.text
    for q
    in test_irds.queries_iter()
}

test_qrels = defaultdict(set)

for qr in test_irds.qrels_iter():

    if qr.relevance > 0:

        test_qrels[
            str(qr.query_id)
        ].add(
            str(qr.doc_id)
        )


print(
    "train queries:",
    len(train_queries)
)

print(
    "test queries:",
    len(test_queries)
)


# ============================================================
# BM25 CANDIDATE GENERATOR
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
    for did in doc_ids
]

bm25 = BM25Okapi(
    tokenized_docs
)


# ============================================================
# TRAIN DATA
#
# IMPORTANT:
#
# Cross-encoder needs positive + negative query-doc pairs.
#
# Positive:
#   qrel docs
#
# Negative:
#   hard negatives from BM25 top results
#   excluding gold relevant docs.
# ============================================================

print()
print("=" * 100)
print("BUILD CROSS-ENCODER TRAIN DATA")
print("=" * 100)

train_rows = []


for qid in tqdm(
    train_qrels.keys()
):

    if qid not in train_queries:
        continue

    query = train_queries[
        qid
    ]

    positives = train_qrels[
        qid
    ]


    # --------------------------------------------------------
    # positive pairs
    # --------------------------------------------------------

    for did in positives:

        if did not in docs:
            continue

        train_rows.append({
            "query": query,
            "document": docs[did],
            "label": 1.0,
        })


    # --------------------------------------------------------
    # BM25 hard negatives
    # --------------------------------------------------------

    scores = np.asarray(
        bm25.get_scores(
            tokenize(query)
        )
    )

    order = np.argsort(
        scores
    )[::-1]

    neg_candidates = []

    for idx in order:

        did = doc_ids[idx]

        if did in positives:
            continue

        neg_candidates.append(
            did
        )

        if len(neg_candidates) >= (
            len(positives)
            *
            NEG_PER_POSITIVE
        ):
            break


    for did in neg_candidates:

        train_rows.append({
            "query": query,
            "document": docs[did],
            "label": 0.0,
        })


random.shuffle(
    train_rows
)


print(
    "train pairs:",
    len(train_rows)
)

print(
    "positive:",
    sum(
        x["label"] == 1
        for x in train_rows
    )
)

print(
    "negative:",
    sum(
        x["label"] == 0
        for x in train_rows
    )
)


train_dataset = Dataset.from_list(
    train_rows
)


# ============================================================
# TRAIN CROSS ENCODER
# ============================================================

print()
print("=" * 100)
print("TRAIN CROSS ENCODER")
print("=" * 100)


if (
    FORCE_TRAIN
    or
    not os.path.exists(
        FINETUNED_MODEL_DIR
    )
):

    ft_model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,
    )


    train_loss = BinaryCrossEntropyLoss(
        ft_model
    )


    args = CrossEncoderTrainingArguments(

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

        fp16=torch.cuda.is_available(),

        logging_steps=20,

        save_strategy="epoch",

        report_to="none",

        seed=SEED,
    )


    trainer = CrossEncoderTrainer(

        model=ft_model,

        args=args,

        train_dataset=(
            train_dataset
        ),

        loss=train_loss,
    )


    trainer.train()


    ft_model.save_pretrained(
        FINETUNED_MODEL_DIR
    )


# ============================================================
# LOAD PRETRAINED / FINETUNED
# ============================================================

print()
print("=" * 100)
print("LOAD CROSS ENCODERS")
print("=" * 100)


pretrained_ce = CrossEncoder(
    BASE_MODEL
)

finetuned_ce = CrossEncoder(
    FINETUNED_MODEL_DIR
)


# ============================================================
# EVALUATION CANDIDATES
#
# BM25 top-K 안에서 cross encoder reranking.
#
# 이게 실제 reranker 사용 방식에 더 가까움.
# ============================================================

all_rows = []


def predict_ce(
    model,
    query,
    candidate_ids,
):

    pairs = [
        [
            query,
            docs[did]
        ]
        for did
        in candidate_ids
    ]

    scores = model.predict(
        pairs,
        batch_size=EVAL_BATCH_SIZE,
        show_progress_bar=False,
    )

    scores = np.asarray(
        scores
    ).reshape(-1)

    return scores


print()
print("=" * 100)
print("TEST CROSS ENCODERS")
print("=" * 100)


for qid in tqdm(
    test_qrels.keys()
):

    query = test_queries[
        qid
    ]

    relevant = test_qrels[
        qid
    ]


    # --------------------------------------------------------
    # BM25 top-K candidate set
    # --------------------------------------------------------

    bm25_scores = np.asarray(
        bm25.get_scores(
            tokenize(query)
        )
    )

    bm25_order = np.argsort(
        bm25_scores
    )[::-1][:TOP_K]

    candidate_ids = [
        doc_ids[i]
        for i in bm25_order
    ]


    # ========================================================
    # pretrained CE
    # ========================================================

    pre_scores = predict_ce(
        pretrained_ce,
        query,
        candidate_ids
    )

    pre_rank_order = np.argsort(
        pre_scores
    )[::-1]

    pre_rank = np.empty_like(
        pre_rank_order
    )

    pre_rank[
        pre_rank_order
    ] = np.arange(
        len(
            pre_rank_order
        )
    )


    for i, did in enumerate(
        candidate_ids
    ):

        all_rows.append({
            "qid": qid,
            "query": query,
            "doc_id": did,
            "retriever": (
                "CrossEncoder_Pretrained"
            ),
            "rank": int(
                pre_rank[i]
            ) + 1,
            "score": float(
                pre_scores[i]
            ),
            "label": int(
                did in relevant
            ),
        })


    # ========================================================
    # finetuned CE
    # ========================================================

    ft_scores = predict_ce(
        finetuned_ce,
        query,
        candidate_ids
    )

    ft_rank_order = np.argsort(
        ft_scores
    )[::-1]

    ft_rank = np.empty_like(
        ft_rank_order
    )

    ft_rank[
        ft_rank_order
    ] = np.arange(
        len(
            ft_rank_order
        )
    )


    for i, did in enumerate(
        candidate_ids
    ):

        all_rows.append({
            "qid": qid,
            "query": query,
            "doc_id": did,
            "retriever": (
                "CrossEncoder_Finetuned"
            ),
            "rank": int(
                ft_rank[i]
            ) + 1,
            "score": float(
                ft_scores[i]
            ),
            "label": int(
                did in relevant
            ),
        })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_rows
)

df.to_csv(
    OUTPUT_DIR
    +
    "/crossencoder_scores.csv",
    index=False
)


# ============================================================
# RANKING METRICS
# ============================================================

ranking_rows = []


for retriever in [
    "CrossEncoder_Pretrained",
    "CrossEncoder_Finetuned",
]:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    for qid, group in tmp.groupby(
        "qid"
    ):

        group = group.sort_values(
            "rank"
        )

        labels = (
            group[
                "label"
            ]
            .to_numpy()
        )


        rr = 0.0

        for rank, label in enumerate(
            labels,
            start=1
        ):

            if label == 1:

                rr = (
                    1.0
                    /
                    rank
                )

                break


        def recall_at(k):

            relevant_total = len(
                test_qrels[
                    str(qid)
                ]
            )

            hit = int(
                labels[:k].sum()
            )

            return (
                hit
                /
                relevant_total
            )


        ranking_rows.append({

            "qid": qid,

            "retriever": (
                retriever
            ),

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
        })


ranking_df = pd.DataFrame(
    ranking_rows
)


ranking_summary = (
    ranking_df
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
print("RANKING METRICS")
print("=" * 100)

print(
    ranking_summary
)


ranking_summary.to_csv(
    OUTPUT_DIR
    +
    "/ranking_metrics.csv"
)


# ============================================================
# SCORE DISTRIBUTION STATS
# ============================================================

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


print()
print("=" * 100)
print("SCORE DISTRIBUTIONS")
print("=" * 100)

print(
    stats
)


stats.to_csv(
    OUTPUT_DIR
    +
    "/score_statistics.csv"
)


# ============================================================
# DISTRIBUTION PNG
# ============================================================

retrievers = [
    "CrossEncoder_Pretrained",
    "CrossEncoder_Finetuned",
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
        "Cross-encoder score"
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

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        n
    )

    output = []


    for threshold in thresholds:

        pred = (
            scores
            >=
            threshold
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
            tp / (tp + fp)
            if tp + fp
            else 1.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
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
            if precision + recall
            else 0.0
        )


        output.append({

            "threshold": float(
                threshold
            ),

            "precision": precision,

            "recall": recall,

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
        tmp[
            "score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    labels = (
        tmp[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )


    sweep = threshold_sweep(
        scores,
        labels
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
            best[
                "threshold"
            ]
        ),

        "precision": (
            best[
                "precision"
            ]
        ),

        "recall": (
            best[
                "recall"
            ]
        ),

        "f1": (
            best[
                "f1"
            ]
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
        "best threshold:",
        best[
            "threshold"
        ]
    )

    print(
        "precision:",
        best[
            "precision"
        ]
    )

    print(
        "recall:",
        best[
            "recall"
        ]
    )

    print(
        "f1:",
        best[
            "f1"
        ]
    )

    print(
        "roc auc:",
        auc
    )


    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    x = (
        sweep[
            "threshold"
        ]
        .to_numpy()
    )

    precision = (
        sweep[
            "precision"
        ]
        .to_numpy()
    )

    recall = (
        sweep[
            "recall"
        ]
        .to_numpy()
    )

    f1 = (
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
        precision,
        label="Precision",
    )

    plt.plot(
        x,
        recall,
        label="Recall",
    )

    plt.plot(
        x,
        f1,
        label="F1",
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

threshold_summary_df.to_csv(
    OUTPUT_DIR
    +
    "/threshold_summary.csv",
    index=False
)


# ============================================================
# TOP-1 SCORE DISTRIBUTION PNG
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
        top1[
            "score"
        ].to_numpy(),
        bins=50,
    )

    plt.xlabel(
        "Top-1 cross-encoder score"
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
# SCORE OVERLAP
# ============================================================

overlap_rows = []


print()
print("=" * 100)
print("SCORE OVERLAP")
print("=" * 100)


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


    perfectly_separable = (
        max_negative
        <
        min_positive
    )


    overlap_amount = (
        max_negative
        -
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
    index=False
)


# ============================================================
# COUNTEREXAMPLES
# ============================================================

counterexample_rows = []


print()
print("=" * 100)
print("EXTREME COUNTEREXAMPLES")
print("=" * 100)


for retriever in retrievers:

    tmp = df[
        df.retriever
        ==
        retriever
    ]


    low_positive = (
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


    high_negative = (
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


    for _, row in low_positive.iterrows():

        print()
        print(
            "score:",
            round(
                float(
                    row[
                        "score"
                    ]
                ),
                4
            )
        )

        print(
            "rank:",
            row[
                "rank"
            ]
        )

        print(
            "query:",
            row[
                "query"
            ]
        )

        print(
            "doc:",
            docs[
                row[
                    "doc_id"
                ]
            ][:400]
        )


        counterexample_rows.append({

            "retriever": (
                retriever
            ),

            "type": (
                "low_score_true_positive"
            ),

            "score": (
                row[
                    "score"
                ]
            ),

            "rank": (
                row[
                    "rank"
                ]
            ),

            "query": (
                row[
                    "query"
                ]
            ),

            "doc_id": (
                row[
                    "doc_id"
                ]
            ),

            "document": (
                docs[
                    row[
                        "doc_id"
                    ]
                ]
            ),
        })


    print()
    print(
        "HIGH-SCORE FALSE POSITIVES"
    )


    for _, row in high_negative.iterrows():

        print()
        print(
            "score:",
            round(
                float(
                    row[
                        "score"
                    ]
                ),
                4
            )
        )

        print(
            "rank:",
            row[
                "rank"
            ]
        )

        print(
            "query:",
            row[
                "query"
            ]
        )

        print(
            "doc:",
            docs[
                row[
                    "doc_id"
                ]
            ][:400]
        )


        counterexample_rows.append({

            "retriever": (
                retriever
            ),

            "type": (
                "high_score_false_positive"
            ),

            "score": (
                row[
                    "score"
                ]
            ),

            "rank": (
                row[
                    "rank"
                ]
            ),

            "query": (
                row[
                    "query"
                ]
            ),

            "doc_id": (
                row[
                    "doc_id"
                ]
            ),

            "document": (
                docs[
                    row[
                        "doc_id"
                    ]
                ]
            ),
        })


pd.DataFrame(
    counterexample_rows
).to_csv(
    OUTPUT_DIR
    +
    "/counterexamples.csv",
    index=False
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
    "[Ranking]"
)

print(
    ranking_summary
)


print()
print(
    "[Threshold]"
)

print(
    threshold_summary_df
    .to_string(
        index=False
    )
)


print()
print(
    "[Overlap]"
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

  crossencoder_scores.csv
  ranking_metrics.csv
  score_statistics.csv
  threshold_summary.csv
  score_overlap.csv
  counterexamples.csv

  crossencoder_pretrained_distribution.png
  crossencoder_finetuned_distribution.png

  crossencoder_pretrained_threshold_curve.png
  crossencoder_finetuned_threshold_curve.png

  crossencoder_pretrained_top1.png
  crossencoder_finetuned_top1.png
""")
