# Beyond-ETRI · PTDTransformer

**A Deviation-Aware Transformer for Multimodal Lifelog-Based Sleep Quality Prediction**

제5회 ETRI 휴먼이해 인공지능 논문경진대회 제출 코드.
스마트폰·웨어러블 12종 센서의 라이프로그로부터 하루 단위 수면 관련 7개 이진 지표
(Q1–Q3 주관적: 수면의 질·피로·스트레스 / S1–S4 객관적 수면 세부 지표)를 예측한다.

- 논문: *PTDTransformer: A Deviation-Aware Transformer for Multimodal Lifelog-Based
  Sleep Quality Prediction* — Chaeyoung Jung, Donghee Kim, Yerin Tak
  (숙명여자대학교 인공지능공학과)
- 모델 설명서: [`docs/MODEL_DESCRIPTION.md`](docs/MODEL_DESCRIPTION.md)
- 리더보드 점수: **0.63265** (5-seed 앙상블 + 확률 보정)

## 핵심 아이디어

각 피처를 **"그 사람의 평소 값(raw)"** 과 **"평소 대비 편차(deviation, 개인
z-score)"** 두 갈래로 분리해 입력하고, 편차를 query·원시 행동을 key/value 로 두는
`DeviationAwareAttention` 으로 융합한 뒤 CLS-token Transformer 로 인코딩한다.
소규모 코호트(10명)에 맞춰 LOSO-CV 평가 + 다중 시드 앙상블 + 확률 보정을 사용한다.

## 저장소 구조

```
├── src/                     모듈화된 파이프라인 (단일 소스)
│   ├── config.py              경로(/data), 하이퍼파라미터, 시드
│   ├── data_loading.py        Parquet 센서 테이블 · 라벨 로드
│   ├── features/              피처 엔지니어링
│   │   ├── common.py            공통 유틸 + 일별 통계 집계
│   │   ├── sleep_proxies.py     screen / light / hr / wifi / usage 수면 프록시
│   │   ├── nested_sensors.py    gps / ambience / ble
│   │   ├── composite.py         복합 · rolling · Q1 전용 피처
│   │   └── deviation.py         라벨 병합 + 개인화 편차 피처
│   ├── dataset.py             시퀀스 데이터셋 (LifelogDataset)
│   ├── model.py               PositionalEncoding, DeviationAwareAttention, PTDTransformer
│   ├── train.py               학습/평가 루프, LOSO-CV
│   └── predict.py             시드 앙상블 학습 + 제출 파일 생성
├── scripts/
│   ├── run_loso_cv.py         LOSO-CV 재현
│   └── make_submission.py     리더보드 제출 파일 재현
├── notebooks/
│   └── PTDTransformer_pipeline.ipynb   end-to-end 노트북 (원본 충실 재현본)
├── experiments/              개발 과정 노트북 (Q/S 분리, XGBoost 등) — 제출물 아님
├── tests/                    합성 데이터 스모크 테스트
├── docs/MODEL_DESCRIPTION.md 모델 설명서 (제출용)
├── data/README.md            /data 입력 레이아웃
└── requirements.txt
```

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

개발 환경: Google Colab (Python 3.11, NVIDIA T4 / CPU 모두 가능). 모델이 작아
CPU에서도 수 분 내 학습된다.

## 데이터 준비

ETRI Lifelog 원시 데이터(Parquet 12개 + CSV 2개)를 `/data` 아래에 배치한다.
파일 목록과 경로 규약은 [`data/README.md`](data/README.md) 참고. 로컬에서는
환경변수로 경로를 바꾼다:

```bash
export DATA_DIR=./data/ch2025
export OUT_DIR=./out
```

## 실행

```bash
# LOSO-CV 성능 재현 (논문 Table III)
python scripts/run_loso_cv.py

# 리더보드 제출 파일 재현 → $OUT_DIR/PTDT_seq7_ensemble_smooth06.csv
python scripts/make_submission.py
```

또는 `notebooks/PTDTransformer_pipeline.ipynb` 를 위에서 아래로 실행.

## 테스트

실제 데이터 없이 합성 데이터로 파이프라인 전체(피처→모델→LOSO→제출)가
오류 없이 도는지 확인한다:

```bash
pip install pytest
pytest -q
```

## LOSO-CV 결과 (논문 Table III)

| Label | Accuracy | Macro-F1 | AUC |
|-------|----------|----------|-----|
| Q1 | 0.450 ± 0.166 | 0.386 ± 0.326 | 0.538 ± 0.121 |
| Q2 | 0.489 ± 0.166 | 0.568 ± 0.269 | 0.506 ± 0.112 |
| Q3 | 0.614 ± 0.124 | 0.712 ± 0.181 | 0.494 ± 0.132 |
| S1 | 0.682 ± 0.172 | 0.798 ± 0.125 | 0.482 ± 0.065 |
| S2 | 0.638 ± 0.219 | 0.756 ± 0.173 | 0.433 ± 0.068 |
| S3 | 0.646 ± 0.236 | 0.756 ± 0.201 | 0.524 ± 0.124 |
| S4 | 0.525 ± 0.209 | 0.604 ± 0.279 | 0.497 ± 0.076 |
| **Macro** | 0.578 ± 0.090 | 0.654 ± 0.099 | 0.496 ± 0.046 |

Macro AUC 는 우연 수준으로, 10명 코호트에서 피험자 독립 예측의 난이도를 보여준다.
앙상블 + 확률 보정이 최종 점수를 0.63265 로 끌어올렸다. 자세한 해석은 모델 설명서 참고.

## 팀

| 이름 | 소속 |
|------|------|
| 정채영 (Chaeyoung Jung) | 숙명여자대학교 인공지능공학과 |
| 김동희 (Donghee Kim) | 숙명여자대학교 인공지능공학과 |
| 탁예린 (Yerin Tak) | 숙명여자대학교 인공지능공학과 |

팀명 `<팀명>` · 팀번호 `<팀번호>`
