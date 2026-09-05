# SciFact: BM25 / Dense / Cross-Encoder Threshold 실험 통합 분석

## 1. 핵심 결론

이번 실험에서 가장 중요한 결론은 아래 3개야.

> **1) Raw BM25 / Dense cosine에 global absolute threshold를 직접 거는 건 불안정하다.**
> **2) Cross-Encoder는 훨씬 더 relevance-oriented한 score를 만들지만, 그래도 perfect global threshold는 불가능했다.**
> **3) Fine-tuning은 ranking과 score separation을 확실히 개선하지만, 자동으로 calibration까지 해결해주지는 않는다.**

즉 실무적으로는:

```text
BM25 / Dense raw score
→ ranking signal로 사용

Cross-Encoder
→ relevance scoring에 더 적합

하지만 최종 binary filtering을 하려면
→ validation 기반 threshold / calibration 필요
```

---

# 2. Retrieval 성능 비교

| Retriever               |        MRR |   Recall@1 |   Recall@5 |  Recall@10 | Recall@100 |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| BM25                    |     0.6236 |     0.5075 |     0.7284 |     0.7740 |     0.8731 |
| Dense Pretrained        |     0.6113 |     0.4823 |     0.7379 |     0.7833 |     0.9250 |
| Dense Fine-tuned        |     0.6275 |     0.5039 |     0.7452 | **0.8264** | **0.9500** |
| CrossEncoder Pretrained |     0.6531 |     0.5486 |     0.7267 |     0.7906 |     0.8731 |
| CrossEncoder Fine-tuned | **0.6974** | **0.5954** | **0.7703** |     0.8074 |     0.8731 |

Dense fine-tuning은 candidate recall 쪽에 강했고, Cross-Encoder fine-tuning은 **top-rank quality**를 크게 끌어올렸다.

특히 CrossEncoder Fine-tuned는:

```text
MRR      0.6531 → 0.6974
Recall@1 0.5486 → 0.5954
Recall@5 0.7267 → 0.7703
```

로 좋아졌다.

즉 Cross-Encoder는 retrieval candidate를 더 잘 “재정렬”하는 역할에서 확실히 강하다.

---

# 3. BM25: ranking은 괜찮지만 global threshold classification은 약함

BM25 relevant/non-relevant score는 상당히 겹쳤다.

```text
Relevant
mean = 41.21
min  = 9.56
max  = 132.96

Non-relevant
mean = 19.84
max  = 90.75
```

즉:

```text
min relevant     = 9.56
max non-relevant = 90.75
```

이라서 하나의 threshold로 완전히 분리할 수 없다.

Best threshold도:

```text
threshold = 55.16
precision = 0.3403
recall    = 0.2211
F1        = 0.2680
ROC-AUC   = 0.8492
```

에 불과했다.

여기서 중요한 포인트는:

```text
ROC-AUC = 0.849
```

로 ranking/discrimination 자체는 나쁘지 않은데,

```text
best threshold F1 = 0.268
```

이라는 점.

즉:

> **BM25 score는 상대적 ordering에는 유용하지만, relevance probability처럼 해석하면 안 된다.**

---

# 4. Dense: cosine도 global threshold로는 여전히 애매함

Dense pretrained는:

```text
Relevant
min = 0.1986
mean = 0.5828

Non-relevant
max = 0.8436
mean = 0.3750
```

이라서 overlap이 매우 컸다.

Best threshold:

```text
threshold = 0.6164
precision = 0.3756
recall    = 0.4589
F1        = 0.4131
ROC-AUC   = 0.8994
```

Fine-tuning 후에는:

```text
ROC-AUC:
0.8994 → 0.9240
```

로 좋아졌지만,

```text
Best F1:
0.4131 → 0.4005
```

로 오히려 약간 떨어졌다.

이게 중요한 이유는:

> **contrastive fine-tuning은 ranking separation을 개선하지만, score를 calibrated probability로 만들어주지는 않는다.**

는 걸 실제로 보여주기 때문이다.

---

# 5. Dense fine-tuning은 무엇을 개선했나

Fine-tuning 전후 평균 score:

```text
Pretrained
Relevant     0.5828
Non-relevant 0.3750

Fine-tuned
Relevant     0.5681
Non-relevant 0.3217
```

positive 평균은 거의 비슷하지만 negative가 더 아래로 내려갔다.

즉:

```text
positive ↔ negative separation
```

은 더 좋아졌다.

그래서 ROC-AUC가:

```text
0.899 → 0.924
```

로 상승했다.

하지만 overlap은 여전히:

```text
Pretrained
min positive = 0.1986
max negative = 0.8436

Fine-tuned
min positive = 0.2233
max negative = 0.8474
```

였기 때문에 perfect global threshold는 불가능했다.

---

# 6. Cross-Encoder는 확실히 한 단계 더 좋아짐

여기서 가장 중요한 결과.

### Pretrained Cross-Encoder

```text
Relevant
mean = 2.2714
std  = 4.2934
min  = -9.0009
max  = 10.1130

Non-relevant
mean = -6.9879
std  = 3.2092
min  = -11.4483
max  = 7.8063
```

### Fine-tuned Cross-Encoder

```text
Relevant
mean = 0.3924
std  = 2.2036
min  = -5.3420
max  = 5.6980

Non-relevant
mean = -4.2898
std  = 1.3456
min  = -9.9173
max  = 3.1713
```

Fine-tuning 후 두 분포가 훨씬 더 compact해졌고 negative 분포가 더 아래로 밀렸다.

---

# 7. Cross-Encoder threshold 성능

## Pretrained

```text
best threshold = 4.4094
precision      = 0.6011
recall         = 0.3844
F1             = 0.4689
ROC-AUC        = 0.9414
```

## Fine-tuned

```text
best threshold = 0.0965
precision      = 0.6653
recall         = 0.5408
F1             = 0.5966
ROC-AUC        = 0.9718
```

이건 꽤 큰 개선이야.

Dense fine-tuned와 비교하면:

| Model                   |    ROC-AUC |    Best F1 |
| ----------------------- | ---------: | ---------: |
| Dense Pretrained        |     0.8994 |     0.4131 |
| Dense Fine-tuned        |     0.9240 |     0.4005 |
| CrossEncoder Pretrained |     0.9414 |     0.4689 |
| CrossEncoder Fine-tuned | **0.9718** | **0.5966** |

즉 relevance thresholding 관점에서는:

```text
BM25
<
Dense
<
Cross-Encoder
<
Fine-tuned Cross-Encoder
```

순서로 좋아졌다고 봐도 된다.

---

# 8. Fine-tuned Cross-Encoder는 global threshold에 훨씬 적합해짐

Fine-tuned CE는 threshold F1이:

```text
0.5966
```

이고 precision / recall도:

```text
precision = 0.665
recall    = 0.541
```

이라서 BM25나 dense보다 훨씬 균형이 좋다.

즉 이 결과는 우리가 앞에서 얘기했던:

```text
raw retriever score
vs
relevance model score
```

차이를 아주 잘 보여준다.

Cross-Encoder는 query-doc pair를 함께 보고 relevance를 직접 학습하기 때문에 binary relevance gate에 더 적합하다.

---

# 9. 하지만 Cross-Encoder도 perfect global threshold는 아님

이게 제일 중요한 caveat.

Pretrained:

```text
min positive = -9.0009
max negative = 7.8063
overlap = 16.8071
```

Fine-tuned:

```text
min positive = -5.3420
max negative = 3.1713
overlap = 8.5133
```

Fine-tuning으로 overlap이 거의 절반 가까이 줄었다.

```text
16.81 → 8.51
```

하지만 여전히:

```text
perfect global threshold possible = False
```

다.

즉:

> **Cross-Encoder도 score가 완전히 calibrated probability가 되는 것은 아니다.**

---

# 10. Fine-tuning 효과는 Cross-Encoder에서 훨씬 명확함

Dense는 fine-tuning 후:

```text
ROC-AUC
0.899 → 0.924

Best F1
0.413 → 0.400
```

이었는데,

Cross-Encoder는:

```text
ROC-AUC
0.941 → 0.972

Best F1
0.469 → 0.597
```

로 둘 다 개선됐다.

즉 Cross-Encoder에서는 fine-tuning이:

```text
ranking quality
+
threshold separability
```

둘 다 좋아지게 만들었다.

이 차이가 중요한 이유는 objective와 architecture 차이 때문이라고 보는 게 자연스럽다.

Dense retriever는 보통:

$$
s(q,d^+) > s(q,d^-)
$$

만 만족하면 되고,

Cross-Encoder binary relevance training은 좀 더 직접적으로:

$$
f(q,d) \rightarrow relevance
$$

를 학습한다.

그래서 thresholding에 더 유리한 score space를 만든다.

---

# 11. Score overlap 비교

| Model            | Min Positive | Max Negative |   Overlap |
| ---------------- | -----------: | -----------: | --------: |
| BM25             |         9.56 |        90.75 |     81.19 |
| Dense Pretrained |        0.199 |        0.844 |     0.645 |
| Dense Fine-tuned |        0.223 |        0.847 |     0.624 |
| CE Pretrained    |       -9.001 |        7.806 |    16.807 |
| CE Fine-tuned    |       -5.342 |        3.171 | **8.513** |

단위가 달라서 overlap absolute magnitude 자체를 모델 간 직접 비교하면 안 돼.

중요한 건 **같은 모델의 before/after**다.

Cross-Encoder는:

```text
16.81 → 8.51
```

로 overlap이 크게 감소했다.

Dense는:

```text
0.645 → 0.624
```

로 감소 폭이 작았다.

---

# 12. 모델별 threshold 적합성

이번 실험 기준으로 정리하면:

```text
BM25 raw score
────────────────────
ranking: O
global relevance threshold: X


Dense cosine
────────────────────
ranking: O
global relevance threshold: △
fine-tuning 후에도 calibration 약함


Cross-Encoder raw logit
────────────────────
ranking: O
threshold: 꽤 가능
하지만 overlap 존재


Fine-tuned Cross-Encoder
────────────────────
ranking: 매우 좋음
threshold: 가장 실용적
그래도 validation/calibration 필요
```

---

# 13. 가장 중요한 실무적 해석

이 결과를 보고:

> "global threshold는 쓰면 안 된다"

라고 결론내리면 잘못이다.

더 정확한 결론은:

> **global threshold의 적합성은 score가 얼마나 relevance-oriented / calibrated 되어 있느냐에 따라 달라진다.**

Raw BM25/cosine은:

```text
score = retrieval ranking signal
```

에 가깝다.

Fine-tuned Cross-Encoder는:

```text
score ≈ relevance signal
```

에 더 가까워진다.

그래서 실무 구조를 이렇게 가져가는 게 자연스럽다.

```text
Query
  ↓
BM25 / Dense
  ↓
top-K candidates
  ↓
Cross-Encoder
  ↓
calibrated relevance score
  ↓
threshold
  ↓
final results
```

---

# 14. Cross-Encoder에도 calibration을 추가하면 더 좋아질 수 있음

현재 CE score는 raw logit이야.

예를 들어 fine-tuned CE에서:

```text
threshold ≈ 0.0965
```

가 best였는데, 이 값 자체는 다른 dataset/domain으로 그대로 가져가면 안 된다.

대신 validation set에서:

```python
raw_logit
    ↓
Platt scaling
or
Isotonic regression
    ↓
P(relevant)
```

로 calibration하면:

```python
if p_relevant >= 0.7:
    keep
```

처럼 훨씬 해석 가능한 global threshold를 만들 수 있다.

---

# 15. 최종 결론

이번 전체 실험을 한 줄로 압축하면:

> **Raw BM25와 dense cosine은 ranking signal로는 강하지만 global relevance threshold로 쓰기에는 score overlap과 query-wise scale variation이 크다. Cross-Encoder는 query-document interaction을 직접 모델링해 score separation과 threshold performance를 크게 개선하며, fine-tuning 후 ROC-AUC 0.972 / F1 0.597까지 향상됐다. 다만 Cross-Encoder조차 positive/negative score overlap이 남으므로, production에서 global threshold를 사용할 경우 validation 기반 calibration이 여전히 필요하다.**

실무 우선순위로 쓰면:

```text
1. Fine-tuned + calibrated Cross-Encoder threshold
   → 가장 추천

2. Fine-tuned Cross-Encoder raw threshold
   → 꽤 실용적

3. Query/domain-specific dense threshold
   → 제한적으로 가능

4. Raw dense cosine global threshold
   → 주의

5. Raw BM25 global threshold
   → 가장 비추천
```

그리고 이번 결과에서 가장 강한 메시지는 **“threshold가 나쁜 게 아니라, threshold를 거는 score의 성격이 중요하다”**는 거야.
<img width="1800" height="1080" alt="crossencoder_pretrained_top1" src="https://github.com/user-attachments/assets/1e0c27dc-7226-4feb-b28f-2940828ce068" />
<img width="1800" height="1080" alt="crossencoder_pretrained_threshold_curve" src="https://github.com/user-attachments/assets/164a14e9-3a81-4e07-9156-4466c328ba80" />
<img width="1800" height="1080" alt="crossencoder_pretrained_distribution" src="https://github.com/user-attachments/assets/255e4634-7e83-4124-a348-568c24c30ecb" />
<img width="1800" height="1080" alt="crossencoder_finetuned_top1" src="https://github.com/user-attachments/assets/6a039e0b-033c-402b-b55b-a53af153e11b" />
<img width="1800" height="1080" alt="crossencoder_finetuned_threshold_curve" src="https://github.com/user-attachments/assets/16c5ecf3-2382-4b63-84cf-4a7682f348b3" />
<img width="1800" height="1080" alt="crossencoder_finetuned_distribution" src="https://github.com/user-attachments/assets/70f55d5e-c75e-4b98-bdf0-bb5a36419105" />

<img width="1600" height="960" alt="scifact_bm25_threshold_curve" src="https://github.com/user-attachments/assets/86e9404c-ad88-4342-becb-8fb0347aa4c4" />
<img width="1600" height="960" alt="scifact_bm25_distribution" src="https://github.com/user-attachments/assets/e358d90f-717c-48ff-b8d1-8c72c8238b41" />
<img width="1600" height="960" alt="scifact_dense_top1" src="https://github.com/user-attachments/assets/9b7d8697-6537-4327-8e43-f148164ced1c" />
<img width="1600" height="960" alt="scifact_bm25_top1" src="https://github.com/user-attachments/assets/74784ed2-e35c-45c0-b513-ccc21d7b8428" />
<img width="1600" height="960" alt="scifact_dense_threshold_curve" src="https://github.com/user-attachments/assets/950c293a-1896-46f2-bbab-ba1ec47adc06" />
<img width="1600" height="960" alt="scifact_dense_distribution" src="https://github.com/user-attachments/assets/910e7bb8-ff93-47a5-afa0-40b1503330be" />


