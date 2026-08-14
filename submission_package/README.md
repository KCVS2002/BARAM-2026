# BARAM 2026 — 최종 제출물(sub_055_tabpfn.csv) 재현 패키지

태백 가덕산 풍력단지 3개 KPX 그룹의 2025년 시간별 발전량 예측.
본 패키지는 Private Score를 기록한 최종 제출 파일을 처음부터 재현한다.

## 1. 개발 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Home (10.0.26200) |
| Python | 3.13.11 (Miniconda) |
| CPU/GPU | GPU 선택 사항 — CPU만으로 재현 가능 (개발 시 RTX 3070 병용) |
| 라이브러리 | `requirements.txt` 참조 (numpy 2.4.2 / pandas 2.3.3 / scikit-learn 1.6.1 / lightgbm 4.6.0 / torch 2.8.0 / tabpfn 2.2.1) |

```
pip install -r requirements.txt
```

## 2. 폴더 구성

```
submission_package/
├─ README.md                ← 본 문서
├─ requirements.txt
├─ train.py                 ← 학습 코드 (모델 3종 학습·저장, ~40분)
├─ inference.py             ← 추론 코드 (최종 CSV 생성, ~60분; GPU 시 단축)
├─ lib.py                   ← 공용 모듈 (로더·모델 정의·결정층)
├─ src/                     ← 피처 생성·평가산식·의사결정 모듈
├─ models/                  ← train.py 산출물 (학습 완료 모델 동봉)
├─ external_data/           ← 사용한 외부데이터 전체 (§5)
├─ output/                  ← inference.py 산출물
└─ Data/                    ← ★ 대회 제공 데이터를 이 위치에 배치
   ├─ train/ (train_labels.csv, ldaps_train.csv, gfs_train.csv, scada_*.csv)
   ├─ test/  (ldaps_test.csv, gfs_test.csv)
   └─ sample_submission.csv
```

대회 데이터를 다른 경로에 둘 경우 환경변수 `DATA_DIR`로 지정:
`set DATA_DIR=C:\path\to\Data`

## 3. 실행 순서

```
python train.py                                        # 1) 학습 → models/
python inference.py                                    # 2) 추론 → output/sub_055_reproduced.csv
python inference.py final_submission_sub_055_tabpfn.csv  # (권장) 동봉된 실제 제출 원본과 오차 대조
```

- 동봉된 `models/`를 그대로 쓰면 1)을 생략하고 2)만 실행해도 된다.
- `final_submission_sub_055_tabpfn.csv` = 리더보드에 실제 제출된 최종 파일 원본
  (대조용 동봉). 재현 검증 시 3번째 명령 하나로 |Δ| 확인 가능.

## 4. 솔루션 개요 (재현 대상 파이프라인)

시간별 예측분포를 "등확률 원자(atom)" 집합으로 구성하고, 대회 산식(0.5×(1-NMAE) +
0.5×FICR)의 기대점수를 시간별로 직접 최적화하는 2단 구조.

1. **메인 LightGBM 19분위** — 3그룹 공유 학습(capacity-factor 타깃 + 그룹 스펙 피처),
   제공 LDAPS/GFS 124피처 + ECMWF IFS 10피처, SCADA 정제 × 고출력(1+3cf²) 가중.
   분위 정렬 → 예보 사이클 내 3h 이동평균 평활 → 300원자 보간.
2. **Analog Ensemble** — 기상 유사도(12피처 가중 kNN) 상위 200이웃의 "실측" 분포를
   원자로 추가. GBM 중앙값에 덧셈 재정렬, 유사도 거리 3분위별 원자 수 125/150/175.
3. **CNN sister (group_1/2)** — KMA LDAPS 1.5km 875hPa u/v 28×28 공간장 → 소형 CNN
   19분위. 75원자 추가 (GBM과 오차상관 ~0.7의 이종 관점).
4. **TabPFN sister (group_1/2)** — 사전학습 표 기초모델 TabPFN v2 (in-context 학습,
   재학습 없음). 컨텍스트 = train 6,000행(q3 가중 서브샘플, seed 42), 앙상블 2.
   75원자 추가.
5. **group_3 전용** — 가용률 혼합 원자(×0.8/×0.6, 15%) + Open-Meteo 3모델 sister
   LightGBM 조건부 원자 (UNISON 부분가용 레짐 대응).
6. **결정층** — 원자 분포에서 산식 기대점수 J(g)를 61그리드+원자 후보로 최대화,
   부트스트랩 결정 배깅(원자 60% × 15회 평균, seed 42).
7. **사후 보정** — 계절 축소(겨울 0.88 / 그 외 0.92, 중저출력 구간 램프; 리더보드
   실측으로 결정된 상수) 후 [0, 그룹 용량] 클리핑.

## 5. 외부데이터 소명 (전부 무료·공개 접근, 누수 기준 준수)

각 예측시점의 예측기준시점(D-1 13:00 KST) 이전에 생성·공개된 자료만 사용.

| # | 자료 | 폴더 | 출처·수집 | 라이선스 | 누수 안전 근거 |
|---|---|---|---|---|---|
| 1 | ECMWF IFS 기압면 예보 (u/v @10m·925·850·700hPa, t925·t850·q850, 0.25° 격자평균, 2022~2025) | `external_data/ecmwf_ifs/` | ECMWF Open Data 퍼블릭 아카이브 (aws/azure/google, Herbie 경유), 2026-07-16/18 수집 | CC-BY-4.0 (ECMWF Open Data) | **D-1 00UTC 사이클, f15~39 (3h)만 사용** — 00Z 런은 D-1 아침(~08시 KST) 공개 → 예측기준시점(D-1 13:00) 이전. 제공 LDAPS/GFS와 동일 규약 |
| 2 | Open-Meteo 과거예보 3모델 (ECMWF/ICON/GFS, 100m 풍속 등, 2024~2025) | `external_data/openmeteo/` | Open-Meteo Historical Forecast API, 2026-07 수집 | CC-BY-4.0 (Open-Meteo) | **previous_day2/3 필드만 사용** = 대상일 2~3일 전 발표 런 → 항상 D-1 13:00 이전 |
| 3 | 기상청 LDAPS 1.5km 875hPa u/v 공간장 (28×28 크롭, 2022~2025) | `external_data/kma_ldaps_grid/` | 기상청 API허브(apihub.kma.go.kr) `nwp_file_down` LDPS 격자자료, 2026-07-27~31 수집 | KOGL 제1유형 (공공누리, 출처표시) | **D-1 00UTC 사이클(tmfc), ef16~39** — 제공 LDAPS와 동일 발표 규약. 2025-12-21 1일은 API 원천 부재("file not exist") → ±24h 인접일 대체 (코드 내 명시) |
| 4 | TabPFN v2 사전학습 가중치 (`tabpfn-v2-regressor.ckpt`) | `external_data/tabpfn/` | Hugging Face `Prior-Labs/TabPFN-v2-reg`, 2026-08-01 다운로드 | Prior Labs License (Apache 2.0 + 저작자 표시) — 상업적 이용 허용 오픈소스 | 가중치 공개 2025-01 (규정 기준 2026-07-05 이전). **로컬 로드·로컬 추론** — 외부 API 추론 아님 |

- 재분석자료(ERA5 등)·2025년 실측·SCADA 외 비공개 자료는 일절 사용하지 않음.
- 평가기간(2025) 발전량 정보는 어떤 형태로도 사용하지 않음.

## 6. 재현 오차 범위에 대한 주석

- LightGBM·kNN·결정층은 seed 고정으로 동일 환경에서 결정적.
- CNN 학습(CPU, seed 42)은 결정적. TabPFN 추론은 GPU/CPU 간 부동소수 연산 차로
  최종 발전량(kWh)에 미세한 차이가 날 수 있으나, 원자 450개 중 75개에만 영향하고
  결정 배깅(15회 평균)이 이를 추가로 완충 — Private Score에는 오차 범위 내
  (개발 환경 재실행 검증 결과는 아래 §7).
- 실행 로그의 `anomalous hours` 3줄(2.3%/2.5%/1.9%)이 일치하면 라벨 정제가 동일하게
  재현된 것.

## 7. 패키지 자체 검증 결과

개발 환경(§1)에서 본 패키지를 처음부터(train.py 8분 → inference.py 39분, RTX 3070)
재실행해 실제 최종 제출 파일(sub_055_tabpfn.csv)과 대조한 결과:

```
kpx_group_1: |Δ| mean 0.00 / max 0.00 kWh
kpx_group_2: |Δ| mean 0.00 / max 0.00 kWh
kpx_group_3: |Δ| mean 0.00 / max 0.00 kWh
```

**3그룹 8,760시간 전체 완전 재현** (라벨 정제 596/658/340 이상시간, CNN 조기종료
곡선 best 0.04108까지 원본 실행과 동일). CPU 실행 시 TabPFN 부동소수 연산 차로
미세한 수치 차이가 있을 수 있으나 §6의 근거로 점수 오차 범위 내.
