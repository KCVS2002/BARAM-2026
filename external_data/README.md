# 외부데이터 출처·재현 기록 (2차 평가 소명용)

> 공통 누수-안전 기준: 대회 예측기준시점 = **D-1 13:00 KST (D-1 04:00 UTC)**.
> 모든 외부 예보 데이터는 이 시점 이전에 생성·공개된 사이클만 사용한다.

---

## 1. Open-Meteo Previous Runs (ECMWF IFS / ICON / GFS)

- **출처 URL**: https://previous-runs-api.open-meteo.com/v1/forecast (문서: https://open-meteo.com/en/docs/previous-runs-api)
- **라이선스**: CC-BY 4.0 (출처표기: "Weather data by Open-Meteo.com"). 누구나 무료 접근 가능.
- **수집 시점**: 2026-07-11
- **수집 스크립트**: `scripts/collect_openmeteo.py` (완전 재현 가능)
- **지점**: 37.28°N, 128.95°E (태백 가덕산 풍력단지 터빈 중심, `Data/info.xlsx` 좌표 기반)
- **모델·커버 기간** (Previous Runs 아카이브 시작일, 실측 프로브 결과):
  - `icon_global`, `gfs_global`: 2024-02~03월부터
  - `ecmwf_ifs025`: 2024년 2분기부터
  - 세 모델 모두 2025년(테스트 기간) 전체 커버
- **변수**: wind_speed_10m/100m, wind_direction_100m, wind_gusts_10m, temperature_2m,
  surface_pressure × 리드타임(previous_day1/2/3)
- **발표시각 근거 (누수 안전)**:
  - `previous_day2`(대상시각 48h 전 예측): 최악의 경우에도 초기화 ≤ D-1 00:00 KST,
    발표는 초기화 후 4~8시간 내 → 항상 예측기준시점(D-1 13:00 KST) 이전. **전 시간대 안전.**
  - `previous_day3`(72h 전): 항상 안전. 예보 추세 피처용.
  - `previous_day1`(24h 전): 대상시각 ≤ D일 13:00 KST 행만 안전.
    **사용 시 반드시 시간대 마스킹** (대상시각 13시 초과 행은 사용 금지 또는 day2로 대체).
- **저장**: `external_data/openmeteo/{model}_prev_runs.csv` (KST, utf-8-sig)

## 2. Copernicus DEM GLO-30 (지형)

- **출처 URL**: https://copernicus-dem-30m.s3.amazonaws.com/ (AWS Open Data,
  https://registry.opendata.aws/copernicus-dem/)
- **라이선스**: ESA Copernicus 무료 공개 (재배포 허용, 출처표기). 누구나 접근 가능.
- **수집 시점**: 2026-07-11
- **파일**: `Copernicus_DSM_COG_10_N37_00_E128_00_DEM.tif` (37–38°N, 128–129°E 타일)
- **누수 안전**: 정적 지형자료 (시간 개념 없음). 위성 관측 기반 2011~2015년 제작 → 학습·평가 전 기간
  이전에 확정된 자료.
- **용도**: 터빈·격자 주변 고도/경사/능선 노출도 등 정적 피처.
- **저장**: `external_data/dem/`

## 3. NOAA GFS 원본 아카이브 (검토 중 — 아직 미수집)

- **출처**: AWS S3 `s3://noaa-gfs-bdp-pds/` (퍼블릭, 인증 불필요)
- **라이선스**: 미국 정부 public domain
- **계획**: D-1 00 UTC 사이클, f016–f039만 (제공 데이터와 동일 기준). Herbie로 변수 subset.
- **수집 변수**: HPBL(경계층높이), SHTFL(현열플럭스), VVEL 850hPa(연직속도) — 파일럿에서
  VVEL이 예측 절대오차와 상관 0.15~0.32 확인 (불확실성 신호).
- **알려진 결함**: 2022-11-30 00Z 런은 원본 아카이브 GRIB 손상("Wrong message length",
  Google 미러도 동일) → 대상일 2022-12-01 피처는 NaN 처리. 전체 1,461일 중 1일.
- **상태**: ✅ 2026-07-12 전 기간 수집 완료 — 946,287행, 1,461일(2022-01-01~2025-12-31 대상일),
  누락은 손상 런 1일(2022-12-01 대상분)뿐. `gfs_aux_20220101_20251231.csv`

## 4. 기상청 API허브 (활용신청 대기)

- **출처**: https://apihub.kma.go.kr (무료 회원가입, API별 활용신청 필요)
- **키**: `.env`의 `KMA_APIHUB_KEY` (커밋/제출물에 포함 금지)
- **대상**: UM 국지(l015, LDAPS 원본) 경량화 다운로드 — 보유기간 2012~현재로 학습기간 전체 커버 예상.
  ※ KIM 국지(L010)는 2026-02부터라 본 대회 기간에 사용 불가.
- **누수 기준**: tmfc = D-1 00 UTC 사이클만, ef=16~39.
- **제약**: 일일 트래픽 할당량 5GB (GRIB ~1MB/건 → 약 4,800건/일). 수집기는 4,600건 도달 시
  자정까지 자동 대기. 전체 67k건 → 약 14일 소요. 연도 우선순위 2024→2025→2023→2022.
- **상태**: ✅ 2026-07-13 지점 API(`nph-um_grib_pt_txt1`, group=UMKR/nwp=N512)로 전 기간 수집 완료.
  1,682,442행 / 35,052 슬롯 (실패 12건=0.03%), 전 기압면 u/v 프로파일.
  유효 레벨: 875hPa부터 (900↓는 지형 채움값). 정합성 검증: 지도 방식과 값 일치(1e-4 이내),
  수집기 시작 시 자동 self-check. `point_profile.csv`, `scripts/collect_kma_point.py`

## 5. NOAA GFS D-2 사이클 (lagged ensemble / 사이클 간 spread)

- **출처**: AWS S3 `s3://noaa-gfs-bdp-pds/` (퍼블릭, 인증 불필요, Herbie 경유)
- **라이선스**: 미국 정부 public domain. 누구나 접근 가능.
- **수집 시점**: 2026-07-23 (스크립트 `scripts/collect_gfs_lag.py`, 완전 재현 가능)
- **내용**: **D-2 00UTC 사이클 f040~f063** (대상일 KST 01~24시 대응), UGRD/VGRD 100m + GUST,
  제공 GFS와 동일 9개 격자 (37.0~37.5N, 128.75~129.25E).
- **누수 안전 근거**: D-2 00UTC 런의 발표는 D-2 04~07UTC — 예측기준시점(D-1 04UTC =
  D-1 13:00 KST)보다 약 하루 전. 전 시간대 안전.
- **용도**: 이전 사이클 예보(lagged ensemble) 및 D-1↔D-2 사이클 간 차이(spread — 예보
  불확실성 신호) 피처. 이론 조사(research/domain_theory.md §2) 후보.
- **상태**: ✅ 전 기간(2022-01-01~2025-12-31 대상일) 수집 완료 — 946,467행,
  35,057/35,064 슬롯 (실패 7건=0.02%). `gfs_lag_d2_20220101_20251231.csv`

## 6. NOAA GFS 광역 격자 (공간 통계 피처)

- **출처·라이선스**: §5와 동일 (AWS S3 `noaa-gfs-bdp-pds`, public domain, Herbie 경유)
- **수집 시점**: 2026-07-24 (`scripts/collect_gfs_grid.py`, 완전 재현 가능)
- **내용**: D-1 00UTC f016~f039 (제공 GFS와 동일 누수 기준), UGRD/VGRD 100m,
  **0.25° 11×11 격자 (36.0~38.5N, 127.5~130.0E, ~275km 도메인)** — 슬롯당 1행
  (풍속 121컬럼 + u/v 평균).
- **용도**: 공간 평균·분산·구배·주성분(PCA) 피처 (Andrade & Bessa 2017,
  research/domain_theory.md §2). 제공 9격자(~50km)가 못 담는 종관 패턴 형태 정보.
- **상태**: ✅ 전 기간 수집 완료 — 35,053/35,064 슬롯 (실패 11건=0.03%).
  `gfs_grid100_20220101_20251231.csv`

## 8. ECMWF IFS 기압면 지점 예보 (메인 GBM 피처 — 최종 제출물 사용)

- **출처**: ECMWF Open Data (aws/azure/google 퍼블릭 아카이브, Herbie `model='ifs',
  product='oper'` 경유). 커버리지 2022-01-25~현재.
- **라이선스**: CC-BY 4.0 (ECMWF Open Data). 누구나 무료 접근 가능.
- **수집 시점**: 2026-07-16 (wave-1: u/v @10m·925·850hPa), 2026-07-18 (wave-2:
  t925·t850·u/v700·q850). 스크립트 `scripts/collect_ifs_point.py`, `collect_ifs_point2.py`
  (완전 재현 가능, (run,fxx) 체크포인트).
- **누수 안전 근거**: **D-1 00UTC 사이클만, f15~39 (3h 간격)** — 00Z 런은 D-1 아침
  (~08시 KST) 공개 → 예측기준시점(D-1 13:00 KST) 이전. 제공 LDAPS/GFS와 동일 규약.
  3h→1h는 피처 단계에서 시간 보간 (`limit=2`).
- **격자**: 0.25° 발전단지 주변 (37.0~37.6N, 128.7~129.3E) 스칼라 풍속 격자 평균.
- **저장**: `external_data/ecmwf_ifs/ifs_point_2022_2025.csv`, `ifs_point2_2022_2025.csv`

## 9. 기상청 LDAPS 1.5km 875hPa 공간장 (CNN sister 입력 — 최종 제출물 사용)

- **출처**: 기상청 API허브 (apihub.kma.go.kr) LDPS 격자 GRIB 다운로드 API.
- **라이선스**: 공공누리(KOGL) 제1유형 (출처표시). 무료 회원가입 후 누구나 접근 가능.
- **수집 시점**: 2026-07-27~31 (3키 병렬). 스크립트 `scripts/collect_ldaps_grid.py`
  (완전 재현 가능, 일 단위 체크포인트).
- **누수 안전 근거**: **tmfc = D-1 00UTC 사이클, ef=16~39** — 제공 LDAPS와 동일 발표 규약.
- **내용**: 875hPa u/v, 발전단지 중심 28×28 크롭(±21km, 1.5km 격자), 일 단위 npz.
- **결측 소명**: 2025-12-21 1일은 API 원천 부재("file not exist" 응답 — 서버측 자료 결번)
  → 추론 시 ±24h 인접일 필드로 대체 (재현 코드에 명시). 그 외 2022년 1일 결측.
- **저장**: `external_data/kma_ldaps_grid/{YYYY-MM-DD}.npz` (1,465파일)

## 7. TabPFN v2 사전학습 모델 (sister 분위 예측기)

- **출처**: Hugging Face `Prior-Labs/TabPFN-v2-reg` (pip 패키지 `tabpfn==2.2.1`이
  자동 다운로드, 로컬 캐시 후 로컬 추론 — 외부 API 추론 아님)
- **라이선스**: Prior Labs License (Apache 2.0 + 저작자 표시 요건) — 오픈소스,
  상업적 이용 허용 → 대회 모델 규칙 충족
- **가중치 공개 시점**: 2025-01 (Nature 2025 논문 동시 공개) — 규정 기준
  2026-07-05 이전 충족. 다운로드: 2026-08-01
- **용도**: 제공 데이터 기반 134피처 → 19분위 sister (in-context 학습, 재학습 없음).
  학습 컨텍스트는 대회 제공 train 라벨/피처만 사용 — 외부 데이터 아님
- **참고**: v3(tabpfn_3) 가중치는 계정 게이트·공개시점 미확인으로 사용하지 않음

## 사용 금지 확인

- ERA5 등 재분석자료: 사용하지 않음 (규칙상 금지).
- 2025년 관측 실측자료: 사용하지 않음.
- 외부 API 원격 모델 추론: 사용하지 않음 (데이터 다운로드만).
