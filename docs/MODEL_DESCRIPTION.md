# 모델 설명서 — PTDTransformer

> 제5회 ETRI 휴먼이해 인공지능 논문경진대회 · 팀 `<팀명>` (팀번호 `<팀번호>`)
>
> PTDTransformer: A Deviation-Aware Transformer for Multimodal Lifelog-Based
> Sleep Quality Prediction — Chaeyoung Jung(정채영), Donghee Kim(김동희),
> Yerin Tak(탁예린), 숙명여자대학교 인공지능공학과

---

## 1. 문제 정의

멀티모달 라이프로그(스마트폰 9종 + 웨어러블 3종 센서)로부터 하루 단위 수면 관련
**7개 이진 지표**를 예측한다.

| 그룹 | 레이블 | 의미 |
|------|--------|------|
| 주관적 (Q) | Q1 / Q2 / Q3 | 수면의 질 / 피로 / 스트레스 |
| 객관적 (S) | S1 / S2 / S3 / S4 | 수면 시간·효율 등 수면 세부 지표 |

- 피험자 10명, 라벨이 존재하는 person-day 450개.
- 평가는 **Leave-One-Subject-Out 교차검증(LOSO-CV)** — 피험자 독립 일반화 성능을
  측정하고, 소규모 코호트에서 흔한 "피험자 정체성 단서 학습(shortcut)"을 드러내기 위함.
- 대회 리더보드 지표는 확률값 기반(Log-Loss 계열)이며 제출은 각 레이블의 확률값을 제출.

## 2. 핵심 아이디어

1. **개인 기준선 대비 편차(personal-baseline deviation)를 명시적으로 분리한다.**
   기존 라이프로그 모델은 하루 행동을 절대값으로만 인코딩한다. 본 모델은 각 피처를
   "그 사람에게 평소 어떤 값인가(raw)"와 "평소 대비 얼마나 벗어났는가(deviation)"
   두 갈래로 나누어 입력한다.
2. **도메인 지식 기반 수제 피처.** 단순 통계 집계를 넘어 수면 맥락을 겨냥한 피처
   (화면 사용 마지막 시각, 취침 전 조도·암전 비율, 심박 기반 수면 구간, 앱 사용
   취침 프록시 등)를 설계한다.
3. **소규모 코호트에 맞춘 평가·앙상블 프로토콜.** LOSO-CV로 일반화를 추정하고,
   최종 제출은 다중 시드 앙상블 + 확률 보정으로 안정화한다.

## 3. 데이터 파이프라인

### 3.1 원시 센서 (12종)

모바일: AC status, activity, ambience, BLE, GPS, light, screen, usage stats, Wi-Fi
웨어러블: heart rate, light, pedometer. 모두 Parquet 형식.

> Wi-Fi 스캔 피처는 추출 함수(`extract_wifi_bedtime_proxy`)를 구현했으나 **최종
> 파이프라인에서는 호출하지 않았다.** 코드에는 남겨 두었다.

### 3.2 일별 피처 엔지니어링

| 단계 | 내용 |
|------|------|
| 기본 집계 | 수치형 센서별 `subject_id × date` 기준 mean/std/min/max/count |
| 센서 전용 | 화면(취침 전/새벽 사용, 마지막 on 시각), 조도(암전 비율, 마지막 밝음 시각), 심박(개인 하위 35% 임계값으로 수면 구간·TST·각성 이벤트 추정), 앱 사용(마지막 사용→취침, 첫 아침 사용→기상, 새벽 소셜/미디어 시간), GPS(이동 비율·마지막 이동 시각·야간 이동), ambience(장면 점수·top-1 비율), BLE(주변 기기 수·RSSI 통계) |
| 복합 피처 | `q__night_phone_burden`, `q2__fatigue_proxy`, `q3__stress_proxy` |
| Rolling | 취침 규칙성 관련 컬럼의 최근 3일·7일 평균 |
| Q1 전용 | `q1_bedtime_regularity`, `q1_sleep_hygiene_score` |

시간대 구분: 저녁 20–23시, 취침 직전 22–23시, 새벽 00–05시, 아침 05–10시.
심박·조도·속도 임계값은 임상 기준이 아닌 정성적으로 튜닝한 휴리스틱이다.

### 3.3 개인화 편차 피처

라벨과 inner join(→ 450 person-day) 후, 결측률 70% 초과 피처를 제거하고 결측값을
`피험자별 median → 전체 median → 0` 순으로 대치한다. 이후 각 피처에 대해

```
dev(x, s, t) = clip( (x_{s,t} − μ_{s,t}) / (σ_{s,t} + ε),  −10, +10 ),  ε = 1e-6
```

를 계산한다. μ, σ 는 **직전 7일**의 rolling 평균·표준편차이며 `shift(1)` 로 당일을
제외해 정보 누수를 막는다.

최종 입력은 `[raw 피처 블록 ∥ deviation 피처 블록]` 으로, 두 블록의 차원이 동일하다.
(논문 기준: 기본 집계 60 → 센서 전용 확장 136 → 복합/rolling/Q1 155 → 결측 필터
146 → 편차 추가 후 총 **292차원(146 raw + 146 deviation)**. 실제 차원은 데이터에
따라 달라질 수 있다.)

## 4. 모델 구조 (PTDTransformer)

입력: `(batch, SEQ_LEN=7, N_FEATURES)` — 예측일 직전 7일의 피처 시퀀스.

```
x = [x_raw ∥ x_dev]                         # 중앙에서 raw / deviation 분리
        │
DeviationAwareAttention                     # raw_proj, dev_proj (Linear+GELU)
   ├─ fused   = Linear([raw_emb ∥ dev_emb])
   ├─ attn    = MultiheadAttention(query=dev_emb, key=value=raw_emb)
   └─ e_t     = LayerNorm(fused + attn)
        │
[CLS] 토큰 prepend  +  sinusoidal Positional Encoding
        │
TransformerEncoder (n_layers 층, GELU, batch_first)
        │
CLS 출력  →  공유 MLP (Linear-GELU-Dropout-LayerNorm)
        │
7개 독립 헤드 (Linear-GELU-Linear) → 로짓 7개  (추론 시 sigmoid)
```

`DeviationAwareAttention` 은 **편차를 query, 원시 행동을 key/value** 로 두어
"평소 대비 벗어난 정도"가 "무엇을 주목할지"를 결정하도록 한다.

### 하이퍼파라미터 (최종 제출: 단일 통합 모델)

| 항목 | 값 |
|------|-----|
| SEQ_LEN | 7 |
| d_model / n_heads / n_layers / d_ff | 64 / 4 / 2 / 128 |
| dropout | 0.1 (LOSO-CV·앙상블 학습), 0.05 (별도 단일 저장 모델) |
| optimizer | AdamW (lr 1e-3, weight_decay 1e-4) |
| scheduler | CosineAnnealingLR (LOSO fold 내) |
| loss | `BCEWithLogitsLoss` |
| epochs / batch_size / patience | 30 / 16 / 7 |
| gradient clipping | max_norm 1.0 |
| 가중치 초기화 | Xavier uniform, bias 0 |
| seed | 42 (기본), 앙상블 [42, 7, 2025, 123, 999] |

> **논문과의 차이.** 논문 Methodology 는 Q 전용·S 전용으로 분리한 두 인스턴스
> (Q: SEQ 3 / d_model 64 / 2층, S: SEQ 7 / d_model 32 / 1층)를 서술한다. 이는
> ablation 과정에서 도출한 설계이며, **실제 리더보드 제출(0.63265)을 만든 실행은
> 위 표의 단일 통합 모델**이다. Q/S 분리·threshold 탐색·pos_weight 등을 실험한
> 노트북은 `experiments/` 에 있다.

## 5. 학습 및 추론

### 5.1 LOSO-CV (성능 추정용)

- 피험자 1명을 test 로 고정, 나머지 중 1명을 순환 규칙으로 validation 으로 사용.
- validation macro AUC 기준 early stopping, best 체크포인트로 test 평가.
- fold별 지표를 평균·표준편차로 집계.

### 5.2 최종 제출 모델

1. 전체 450 person-day 로 시드 5개(`[42, 7, 2025, 123, 999]`) 각각 30 epoch 학습.
2. 제출 대상 각 행에 대해 직전 7일 시퀀스를 구성(부족하면 0 패딩)하고 5개 모델의
   sigmoid 확률을 평균.
3. **확률 보정**: `p_out = p · 0.6 + 0.2` — 예측을 [0.2, 0.8] 범위로 수축시켜
   과확신을 억제(Log-Loss 계열 지표에서 안정적 이득).
4. `PTDT_seq7_ensemble_smooth06.csv` 로 저장.

## 6. 결과 (LOSO-CV, 논문 Table III)

| Label | Accuracy | Macro-F1 | AUC |
|-------|----------|----------|-----|
| Q1 | 0.450 ± 0.166 | 0.386 ± 0.326 | **0.538 ± 0.121** |
| Q2 | 0.489 ± 0.166 | 0.568 ± 0.269 | 0.506 ± 0.112 |
| Q3 | 0.614 ± 0.124 | 0.712 ± 0.181 | 0.494 ± 0.132 |
| S1 | 0.682 ± 0.172 | 0.798 ± 0.125 | 0.482 ± 0.065 |
| S2 | 0.638 ± 0.219 | 0.756 ± 0.173 | 0.433 ± 0.068 |
| S3 | 0.646 ± 0.236 | 0.756 ± 0.201 | 0.524 ± 0.124 |
| S4 | 0.525 ± 0.209 | 0.604 ± 0.279 | 0.497 ± 0.076 |
| **Macro** | 0.578 ± 0.090 | 0.654 ± 0.099 | 0.496 ± 0.046 |

- Macro AUC 0.496 은 사실상 우연 수준 — 10명 코호트에서 피험자 독립 예측의
  본질적 난이도를 보여준다. 개별 아키텍처 비교는 보수적으로 해석해야 한다.
- Q1 은 최고 AUC(0.538)이나 최저 Macro-F1(0.386) — 클래스 불균형에서 흔한 괴리.
- **앙상블 + 확률 보정**이 가장 견고한 이득을 주어, 최종 리더보드 점수는
  **0.63265** 로 상승했다.

## 7. 한계

- 피험자 10명·수백 person-day 로 통계적 검정력·일반화가 제한적.
- 여러 센서 전용 피처가 임상 검증되지 않은 휴리스틱 임계값에 의존.
- Wi-Fi 스캔은 추출 함수를 구현했으나 최종 피처셋에서 제외.
- 단일 대회 데이터·단일 코호트-연도의 결과.

## 8. 재현 방법

```bash
pip install -r requirements.txt
# 원시 데이터(Parquet 12개 + CSV 2개)를 /data 아래 배치 (data/README.md 참고)

# LOSO-CV 재현
python scripts/run_loso_cv.py

# 리더보드 제출 파일(PTDT_seq7_ensemble_smooth06.csv) 재현
python scripts/make_submission.py
```

로컬에서는 `DATA_DIR`, `OUT_DIR` 환경변수로 경로를 바꿀 수 있다.
동일 로직의 end-to-end 노트북: `notebooks/PTDTransformer_pipeline.ipynb`.

**개발 환경**: Google Colab (Ubuntu 22.04, Python 3.11, NVIDIA T4 / CPU 모두 가능).
라이브러리 버전은 `requirements.txt` 참고 (numpy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.2, torch 2.5.1, pyarrow 17.0.0).

## 9. 사전 학습 모델

사용하지 않음. 모든 가중치는 위 데이터로 처음부터 학습한다.
