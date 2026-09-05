<img width="1600" height="960" alt="scifact_dense_distribution" src="https://github.com/user-attachments/assets/ddcc38da-dfa1-42cd-8f1a-122c7a94f4bf" />
# SciFact 실험 결과 분석

## 결론

이번 실험은 꽤 명확해.

> **BM25 raw score나 dense cosine에 하나의 global absolute threshold를 걸면 relevance를 안정적으로 분리하기 어렵다.**

더 중요한 건 **fine-tuning 이후에도 이 문제가 사라지지 않았다는 점**이야.

Dense fine-tuning은 retrieval 성능과 score discrimination 자체는 좋아졌지만, `cosine > 특정 값이면 relevant` 같은 형태로 완벽히 분리되지는 않았다. 

---

## 1. Retrieval 성능

| Retriever        |        MRR |   Recall@1 |   Recall@5 |  Recall@10 | Recall@100 |
| ---------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| BM25             |     0.6236 | **0.5075** |     0.7284 |     0.7740 |     0.8731 |
| Dense Pretrained |     0.6113 |     0.4823 |     0.7379 |     0.7833 |     0.9250 |
| Dense Fine-tuned | **0.6275** |     0.5039 | **0.7452** | **0.8264** | **0.9500** |

Fine-tuning 이후 dense는 특히 deeper retrieval에서 개선이 분명하다. Recall@10은 `0.7833 → 0.8264`, Recall@100은 `0.925 → 0.95`로 상승했고 MRR도 `0.6113 → 0.6275`로 좋아졌다. 

흥미로운 건 Recall@1만 보면 BM25가 `0.5075`로 fine-tuned dense의 `0.5039`보다 아주 약간 높다는 점이다. 즉 BM25는 lexical exact-match가 강한 SciFact에서 최상위 1개 후보는 여전히 강하지만, 후보 수가 늘어날수록 fine-tuned dense가 더 많은 relevant 문서를 회수한다.

---

# 2. BM25 score는 query마다 scale이 심하게 달라짐

BM25 relevant score 분포:

```text
Relevant
mean = 41.21
min  = 9.56
25%  = 26.12
50%  = 36.98
75%  = 51.32
max  = 132.96
```

반면 non-relevant:

```text
Non-relevant
mean = 19.84
min  = 4.84
max  = 90.75
```

즉:

```text
minimum relevant = 9.56
maximum irrelevant = 90.75
```

이다. 

이 자체로 global threshold의 문제가 거의 끝난다.

예를 들어:

```text
threshold = 10
```

이면 score 9.56짜리 실제 relevant는 제거된다.

그런데 threshold를 9 정도로 낮추면 score가 20, 30, 50, 심지어 90인 non-relevant가 대거 통과한다.

실제로 score overlap은:

```text
BM25
min positive = 9.5579
max negative = 90.7471

overlap = 81.1892
perfect global threshold possible = False
```

였다. 

---

## 3. BM25 threshold sweep도 이걸 그대로 보여줌

BM25에서 F1을 최대화하는 global threshold는 약:

```text
threshold = 55.16

precision = 0.3403
recall    = 0.2211
F1        = 0.2680
```

밖에 안 된다. 

이게 중요한 이유는 BM25의 ROC-AUC는:

```text
0.8492
```

로 꽤 괜찮기 때문이다.

즉:

```text
ranking ability        = 꽤 좋음
absolute classification = 나쁨
```

이다.

이건 BM25를 이해할 때 중요한 차이야.

BM25는:

```text
이 query에서 doc A가 doc B보다 좋은가?
```

에는 꽤 유용하지만,

```text
score가 50 이상이면 relevant인가?
```

라는 의미의 calibrated score는 아니다.

---

# 4. 실제 BM25 반례가 상당히 극단적임

실제 true positive인데 BM25가:

```text
9.5579
9.9861
10.8139
11.4198
...
```

밖에 안 나오는 문서들이 존재한다. 예를 들어 `CCL19 is absent within dLNs.`의 실제 relevant 문서가 BM25 `9.9861`에 불과했다. 

반대로 non-relevant인데:

```text
90.7471
89.3378
87.4138
85.5875
...
```

까지 올라간다. 

즉 실제 데이터에서:

```text
relevant     = 9.56
non-relevant = 90.75
```

라는 역전 사례가 존재한다.

이건 synthetic example보다 훨씬 강한 증거야.

---

# 5. Dense pretrained도 똑같은 문제가 있음

Pretrained dense의 score 분포:

```text
Relevant
mean = 0.5828
min  = 0.1986
median = 0.6015
max  = 0.8843

Non-relevant
mean = 0.3750
max  = 0.8436
```

이다. 

즉:

```text
true positive minimum = 0.1986
false positive maximum = 0.8436
```

이다.

우리가 앞에서 얘기했던:

> "그럼 threshold를 0.3으로 낮추면 되잖아?"

에 대한 실제 benchmark 답이 여기서 나온다.

---

# 6. threshold=0.3은 recall은 높지만 precision이 사실상 박살남

Pretrained dense에서:

| threshold |  Precision |     Recall |     F1 |
| --------: | ---------: | ---------: | -----: |
|       0.1 |     0.0105 |     1.0000 | 0.0209 |
|       0.2 |     0.0107 |     0.9968 | 0.0211 |
|   **0.3** | **0.0126** | **0.9810** | 0.0248 |
|       0.4 |     0.0256 |     0.9146 | 0.0499 |
|       0.5 |     0.0990 |     0.7152 | 0.1739 |
|       0.6 |     0.3225 |     0.5032 | 0.3931 |
|       0.7 |     0.5217 |     0.1899 | 0.2784 |
|       0.8 |     0.7273 |     0.0253 | 0.0489 |



`0.3`이면 recall은 98.1%라 거의 다 살린다.

문제는:

```text
TP = 310
FP = 24,376
```

이라는 것.

즉 **정답 거의 다 살리려고 threshold를 낮추면 오답도 거의 다 통과한다.**

그래서 단순히:

```python
if cosine >= 0.3:
    keep
```

는 사실상 filtering 역할을 거의 못 한다.

---

# 7. 그런데 threshold를 높이면 recall이 무너짐

`threshold = 0.6`에서는:

```text
precision = 0.3225
recall    = 0.5032
```

이고,

`threshold = 0.7`에서는:

```text
precision = 0.5217
recall    = 0.1899
```

까지 떨어진다. 

즉 매우 전형적인 trade-off:

```text
threshold ↓
→ recall ↑
→ FP 폭증

threshold ↑
→ precision ↑
→ FN 폭증
```

이 그래프에서 그대로 나타난다.

---

# 8. 실제 dense counterexample이 매우 강함

Pretrained dense의 실제 low-score true positive:

```text
CCL19 is absent within dLNs.
cosine = 0.1986

Venules have a larger lumen diameter than arterioles.
cosine = 0.2320

Arterioles have a larger lumen diameter than venules.
cosine = 0.2456
```

등이 존재한다. 

즉 우리가 찾으려고 했던:

> **“cosine이 0.2 언저리인데 실제 relevant인 케이스”**

가 공개 benchmark에서 실제로 나왔다.

반대로 non-relevant인데:

```text
0.8436
0.8384
0.8270
0.7959
...
```

같이 매우 높은 cosine을 가지는 문서도 존재한다. 

즉 실제로:

```text
relevant     = 0.1986
non-relevant = 0.8436
```

이 가능하다.

따라서 global cosine threshold 하나로는 둘을 분리할 방법이 없다.

---

# 9. Fine-tuning하면 어떻게 변했나

Fine-tuning은 꽤 재미있는 결과를 만들었다.

Score distribution:

```text
Dense Fine-tuned

Non-relevant mean = 0.3217
Relevant mean     = 0.5681
```

Pretrained는:

```text
Non-relevant mean = 0.3750
Relevant mean     = 0.5828
```

였다. 

즉 fine-tuning 후 가장 크게 바뀐 건:

```text
negative score가 아래로 밀림
```

이다.

positive 평균은 오히려:

```text
0.5828 → 0.5681
```

로 조금 낮아졌지만,

negative는:

```text
0.3750 → 0.3217
```

로 더 크게 내려갔다.

그래서 **positive-negative separation 자체는 좋아졌다.**

---

# 10. 실제로 ROC-AUC는 크게 상승

Dense pretrained:

```text
ROC-AUC = 0.8994
```

Fine-tuned:

```text
ROC-AUC = 0.9240
```

로 개선됐다. 

즉 학습으로:

> “relevant를 irrelevant보다 위에 놓는 능력”

은 확실히 좋아졌다.

이건 retrieval metric 개선과도 일치한다.

---

# 11. 그런데 threshold F1은 오히려 약간 떨어짐

재미있는 부분이다.

```text
Dense_Pretrained
best threshold = 0.6164
Precision = 0.3756
Recall    = 0.4589
F1        = 0.4131

Dense_Finetuned
best threshold = 0.5799
Precision = 0.3300
Recall    = 0.5093
F1        = 0.4005
```



즉:

```text
ROC-AUC:
0.899 → 0.924 ↑

Retrieval Recall@10:
0.783 → 0.826 ↑

하지만 best threshold F1:
0.413 → 0.400 ↓
```

이게 아주 중요한 결과야.

---

# 12. 왜 학습했는데 threshold classification은 안 좋아졌나

이유는 학습 objective 때문이라고 보는 게 자연스럽다.

이번 fine-tuning은 contrastive retrieval training이라 기본 목적은:

$$
s(q,d^+) > s(q,d^-)
$$

이다.

즉:

```text
positive가 negative보다 높게
```

만들면 된다.

반면 우리가 threshold로 원하는 건:

$$
s(q,d) > T
\iff
relevant
$$

이다.

이 둘은 완전히 다른 목표다.

그래서 fine-tuning 이후 ranking separation은 좋아졌지만 score 자체가 relevance probability로 calibration되지는 않았다.

---

# 13. Fine-tuning 후에도 score overlap은 엄청 큼

결과:

```text
Pretrained:
min positive = 0.1986
max negative = 0.8436
overlap      = 0.6450

Fine-tuned:
min positive = 0.2233
max negative = 0.8474
overlap      = 0.6241
```



overlap 자체는 조금 줄었다:

```text
0.645 → 0.624
```

하지만 여전히 매우 크다.

따라서 fine-tuning 후에도:

```text
perfect global threshold possible = False
```

다.

---

# 14. Fine-tuned에서도 cosine 0.22짜리 실제 정답이 있음

Fine-tuned model에서도:

```text
NOX2-independent pathways...
relevant score = 0.2233

single flash-evoked ERG...
relevant score = 0.2249

ML-SA1...
relevant score = 0.2319
```

같은 사례가 나온다. 

반대로 non-relevant인데:

```text
0.8474
0.8411
0.8303
...
```

까지 나온다. 

즉 학습했다고 해도:

```text
cosine ≈ confidence
```

가 되는 게 아니다.

---

# 15. Fine-tuning 후 fixed threshold 결과

Fine-tuned dense:

| Threshold |  Precision | Recall |         F1 |
| --------: | ---------: | -----: | ---------: |
|       0.1 |     0.0108 | 1.0000 |     0.0214 |
|       0.2 |     0.0111 | 1.0000 |     0.0220 |
|       0.3 |     0.0187 | 0.9599 |     0.0368 |
|       0.4 |     0.0617 | 0.8549 |     0.1151 |
|       0.5 |     0.1792 | 0.6759 |     0.2833 |
|       0.6 | **0.3540** | 0.4414 | **0.3929** |
|       0.7 |     0.5575 | 0.1944 |     0.2883 |
|       0.8 |     0.8000 | 0.0370 |     0.0708 |



여기서도 threshold `0.3`은:

```text
Recall ≈ 96%
Precision ≈ 1.9%
```

라서 practical filter로는 너무 permissive하다.

---

# 16. 이 그래프들이 말하는 것

### Top-1 score distribution

BM25 top-1 score는 대략 `10~130`까지 매우 넓게 퍼져 있다.

즉 같은 "top-1"인데도 query에 따라 raw BM25 scale이 매우 다르다.

Dense top-1도 대략:

```text
0.33 ~ 0.88
```

까지 상당히 넓게 퍼져 있다.

따라서:

```text
top-1 cosine 자체도 query마다 절대 scale이 동일하지 않음
```

을 보여준다.

---

## Relevant vs Non-relevant histogram

가장 중요한 그림은 이거야.

Dense에서 relevant가 우측으로 이동해 있고 non-relevant는 좌측에 많이 몰려 있지만, 중간 `0.35~0.60` 영역에서 상당히 겹친다.

BM25도 마찬가지로 relevant 평균은 높지만 distribution overlap이 매우 크다.

즉:

> **score가 useful하지 않은 게 아니라, score 하나로 binary relevance를 완벽히 결정하기 어렵다는 것.**

---

# 17. 이번 실험에서 가장 중요한 구분

이번 결과를 보고:

> "threshold는 쓰면 안 된다"

라고 결론 내리면 너무 강하다.

정확한 결론은:

> **raw retrieval score에 하나의 fixed global threshold를 두는 것은 retrieval relevance filtering에 취약하다.**

threshold 자체는 충분히 쓸 수 있어.

예를 들어 labeled validation set에서:

```text
query features
BM25
dense cosine
rank
top1-top2 margin
query length
reranker score
```

등을 넣어서:

$$
P(relevant|q,d)
$$

를 학습한 뒤:

```python
if p_relevant > 0.8:
    keep
```

하는 건 전혀 다른 얘기다.

---

# 18. 이번 실험의 가장 강한 증거 3개

### ① BM25

```text
Relevant minimum     = 9.56
Non-relevant maximum = 90.75
```



즉 raw score가 9인 정답과 90인 오답이 동시에 존재.

---

### ② Dense pretrained

```text
Relevant minimum     = 0.1986
Non-relevant maximum = 0.8436
```



그래서 `threshold=0.2`, `0.3`, `0.4` 중 무엇을 골라도 trade-off가 발생.

---

### ③ Fine-tuning도 문제를 제거하지 못함

```text
Dense fine-tuned

ROC-AUC 0.899 → 0.924
Recall@10 0.783 → 0.826

하지만

min relevant     = 0.223
max non-relevant = 0.847
```

즉 **ranking은 개선되지만 score calibration은 자동으로 해결되지 않는다.** 

---

# 최종 정리

이번 실험 결과를 한 문장으로 요약하면:

> **BM25와 dense cosine은 relevance ranking signal로는 유용하지만 calibrated probability가 아니므로, raw absolute threshold 하나로 relevance filtering을 하면 false positive와 false negative를 동시에 피하기 어렵다. Contrastive fine-tuning은 ranking discrimination과 retrieval recall을 개선하지만 global score calibration 문제까지 자동으로 해결하지는 않는다.**

실무적으로는 이렇게 보는 게 가장 맞아.

```text
Bad
─────────────────────────
BM25 > 30
cosine > 0.5
       ↓
KEEP


Better
─────────────────────────
retrieve top-K
       ↓
reranker / relevance model
       ↓
P(relevant | q,d)
       ↓
calibrated threshold
       ↓
KEEP / DROP
```

<img width="1600" height="960" alt="scifact_bm25_threshold_curve" src="https://github.com/user-attachments/assets/86e9404c-ad88-4342-becb-8fb0347aa4c4" />
<img width="1600" height="960" alt="scifact_bm25_distribution" src="https://github.com/user-attachments/assets/e358d90f-717c-48ff-b8d1-8c72c8238b41" />
<img width="1600" height="960" alt="scifact_dense_top1" src="https://github.com/user-attachments/assets/9b7d8697-6537-4327-8e43-f148164ced1c" />
<img width="1600" height="960" alt="scifact_bm25_top1" src="https://github.com/user-attachments/assets/74784ed2-e35c-45c0-b513-ccc21d7b8428" />
<img width="1600" height="960" alt="scifact_dense_threshold_curve" src="https://github.com/user-attachments/assets/950c293a-1896-46f2-bbab-ba1ec47adc06" />
<img width="1600" height="960" alt="scifact_dense_distribution" src="https://github.com/user-attachments/assets/910e7bb8-ff93-47a5-afa0-40b1503330be" />



`0.3`에서는 pretrained 기준 **recall 98.1%지만 precision 1.26%**, fine-tuned에서도 **recall 96.0%지만 precision 1.87%**밖에 안 된다. 즉 **threshold를 낮추면 정답은 살지만 filter 자체가 거의 의미 없어지는 것**이 실제 공개 데이터에서 확인됐다. 
