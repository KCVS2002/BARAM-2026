### 1. (그래픽) 초단기강수예측 조회

#### **API 활용신청**

https://apihub.kma.go.kr/api/typ03/cgi/dfs/nph-qpf_ana_img?eva=1&tm=202212221350&qpf=B&ef=360&map=HR&grid=2&legend=1&size=600&zoom_level=0&zoom_x=0000000&zoom_y=0000000&stn=108&x1=470&y1=575&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| eva | 평가기준 | 2:0.1mm, 3:1mm, 4:5mm |
| tm | 조회할 자료시각 |  |
| qpf | API종류 | K(초단기모델), M(레이더외삽), B(융합예측) |
| ef | 예측시간 | +0,+1,,, |
| map | 사용할 지도코드 |  |
| grid | 격자크기(km) |  |
| legend | 범례표시(1) 여부 |  |
| size | 이미지크기(픽셀) |  |
| stn | 시간간격 |  |
| x1 | 점 표출 |  |
| y1 | 점 표출 |  |
| authKey | 인증키 | 발급된 API 인증키 |





### 2.수치예보모델


| **개요** | (전구) 전지구 예보모델(GDAPS, Global Data Assimilation and Prediction System)은 전지구 날씨 예측, 동네예보, 중기예보 등을 목적으로 영국 통합모델(UM)과 한국형수치모델(KIM)을 기반 구축된 수치예보시스템입니다.
6시간 주기 순환예측에 필요한 각종 배경장을 생산하기 위한 일 4회의 15시간 예측을 수행하며, 일 1회 06UTC에는 해수면온도, 해빙자료 및 동서평균 오존량을 갱신하기 위한 배경장 갱신과정을 별도로 수행합니다.

(지역) GDAPS 예측자료에서 동아시아 영역의 자료만 추출하여 생산, 제공하고 있습니다.

(국지) 국지예보모델(LDAPS, Local Data Assimilation and Prediction System)은 3시간 간격으로 전지구 예보모델로부터 경계장을 제공받아 1일 4회 예측을 수행하는 수치예보 시스템입니다.
3차원 변분자료 동화 기법을 이용하여 각각의 자체 분석-예측 순환 체계로 운영되고 있으며, 주로 한반도 날씨 예측에 활용됩니다. |
| --- | --- |
| **요 소** | (전구, 지역) 등압면, 단일면 요소
(국지) 등압면, 단일면, 모델면 |
| **해상도** | 기간에 따라 해상도가 상이함(20xx.xx.xx. 기준)

(전구)
ㆍ (공간) 수평분해능 : 10km/ 연직층수 : 70층
ㆍ (시간) 288시간 예측(00, 12UTC)/ 87시간 예측(06, 18UTC)

(지역)
ㆍ (공간) 수평분해능 : 12km/ 연직층수 : 70층
ㆍ (시간) 87시간 예측(00, 06, 12, 18UTC)

(국지)
ㆍ (공간) 수평분해능 : 1.5km/ 연직층수 : 70층
ㆍ (시간) 48시간 예측(00, 06, 12, 18UTC)
* 예측시간: 24시간(2012.5.15. ~), 36시간(2013.4.29. ~), 48시간(2019.5.28. ~) |
| **보유기간** | (전구) 2011년 5월 ~ 현재
(지역) 2010년 3월 ~ 현재
(국지) 2012년 5월 ~ 현재
※ 보존정책에 따라 데이터 조회기간은 다를 수 있음 |
| **생산주기** | 일 4회(00, 06, 12, 18UTC) |

### 1. 수치모델 경량화 다운로드(예측시간+변수+고도별)

수치모델 GRIB 변수 코드 등 참고자료

KIM 지역 및 국지 조회변수 코드 참고자료%EC%A7%80%EC%97%AD%EB%B0%8F%EA%B5%AD%EC%A7%80%EC%A1%B0%ED%9A%8C%EB%B3%80%EC%88%98%EB%A6%AC%EC%8A%A4%ED%8A%B8.pdf)

#### 1.1.1 UM모델 다운로드

https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php?nwp=g128&sub=pres&vars=tmpr&pres=850&tmfc=2021081012&ef=24&dataType=TEXT&authKey={인증키입력}

#### 1.1.2 KIM모델 다운로드

https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php?nwp=l010&sub=pres&vars=tmpr&pres=850&tmfc=202603101200&ef=24&dataType=TEXT&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| nwp | KIM모델 | r030(KIM지역), l010(KIM국지) |
| nwp | UM모델 | g128(UM전구), g768(UM전구), g512(UM전구),
g120(UM지역,20190401~), r120(UM지역,~20190331), l015(UM국지)
※ UM모델은 2026.3.31. 생산종료 |
| sub | 파일구분 | "pres(등압면), unis(단일면) ... UM모델만
isen(등온위면) ... UM모델만
없으면 해당모델 전체" |
| tmfc | 기준시각 | 년월일, 년월일시(UTC) |
| ef | 예측시각 | 기준시각에서부터의 예측시간(0~) |
| dataType | 저장포맷 | GRIB, BIN(이진), TEXT(csv) |
| authKey | 인증키 | 발급된 API 인증키 |

### 2. 한국형수치예보모델(KIM) 표준화 자료 조회(NC)

한국형 수치모델(KIM) 변수정보 참고자료%20%EB%B3%80%EC%88%98%EC%A0%95%EB%B3%B4.pdf)

#### 2.1.1 전체영역

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_xy_txt2_std?group=KIMG&nwp=NE57&data=U&name=t2m&map=F&tmfc=2026061000&hf=0&disp=A&help=1&level=0&authKey={인증키입력}

#### 2.1.2 일부 격자영역

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_xy_txt2_std?group=KIMR&nwp=R030&data=P&name=U&map=S&sub=300,300,600,500&sm=0&tmfc=2026061000&hf=0&disp=A&help=1&level=500&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| group | 모델 구분 | KIMG(전구), KIMR(지역), KIML(국지) |
| nwp | 모델기반 종류 | NE57(전구, '26.1.19.~), R030(지역, '26.2.9.~), L010(국지, '26.2.9.~) |
| data | 자료 종류 | P(등압면), U(단일면) |
| name | 변수명 | 한국형 수치모델(KIM) 변수정보 참고 / 대소문자 주의 |
| level | 고도 | 등압면자료의 경우, 등압면고도(hPa)로 입력
단일면의 경우도 토양면처럼 층이 있는 경우는 값이 있어야 됨
격자점 자료 조회의 경우에 level값이 있으면 해당 고도값만 표출, 없으면 각 고도의 값을 모두 표출 |
| map | 사용 영역 | F: 자료 전체 영역
S: 일부 격자 영역만 추출 |
| sub | 격자영역 | map=S 인 경우, 추출할 격자 영역을 표시
표시방법은 sub=x_min,y_min,x_max,ymax
왼쪽아래 격자점 [x_min, y_min]에서 오른쪽 위 격자점 [x_max, y_max] 까지를 의미
최소 격자점은 [1,1] 임 |
| tmfc | 분석시간 | 년월일시(UTC) / 조회기간은 영역별 상이 |
| hf | 예측시간 | 시간(hour) (예: 24 = +24H) |

#### 2.2.1 임의 격자점의 자료

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std?group=KIMG&nwp=NE57&data=P&name=T,q&tmfc=2026061000&hf=0&disp=A&help=1&X=50&Y=100&authKey={인증키입력}

#### 2.2.2 임의 위경도의 자료

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std?group=KIML&nwp=L010&data=P&name=U,W&tmfc=2026061000&hf=0&disp=A&help=1&lat=38&lon=150&disp=A&help=1&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| group | 모델 구분 | KIMG(전구), KIMR(지역), KIML(국지) |
| nwp | 모델기반 종류 | NE57(전구, '26.1.19.~), R030(지역, '26.2.9.~), L010(국지, '26.2.9.~) |
| data | 자료 종류 | P(등압면), U(단일면) |
| name | 변수명 | 한국형 수치모델(KIM) 변수정보 참고 / 대소문자 주의 |
| level | 고도 | 등압면자료의 경우, 등압면고도(hPa)로 입력
단일면의 경우도 토양면처럼 층이 있는 경우는 값이 있어야 됨
격자점 자료 조회의 경우에 level값이 있으면 해당 고도값만 표출, 없으면 각 고도의 값을 모두 표출 |
| map | 사용 영역 | S: 일부 격자 영역만 추출 |
| X, Y | 격자 | 격자점 |
| lat, lon | 위경도 | 위경도 |
| tmfc | 분석시간 | 년월일시(UTC) / 조회기간은 영역별 상이 |
| hf | 예측시간 | 시간(hour) (예: 24 = +24H) |

### 3. 한국형수치예보모델(KIM) 자료 조회(GRIB)

#### 3.1.1 해당 고도의 2차원 단일면 자료

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_xy_txt1?group=KIMR&nwp=r030&data=U&varn=2002&level=0&tmfc=2026030100&hf=0&disp=A&authKey={인증키입력}

#### 3.1.2 해당 고도의 2차원 등압면 자료

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_xy_txt1?group=KIML&nwp=l010&data=P&varn=2002&level=800&tmfc=2026030100&hf=48&disp=A&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| group | 모델 | KIMR(지역), KIML(국지) |
| nwp | 종류 | r030(지역), l010(국지) |
| data | 자료종류 | P(등압면), U(단일면) |
| varn | 변수종류 | D*100000 + C*1000 + P
- D : GRIB.sec0.discipline : 자료의 종류
- C : GRIB.sec4.parameter category : 변수 그룹
- P : GRIB.sec4.parameter number : 변수 번호
격자점 자료 조회의 경우에 level값이 있으면 해당 고도값만 표출, 없으면 각 고도의 값을 모두 표출
예) 850hPa -> level=850으로 입력 |
| tmfc | 분석시간 | 연월일시(UTC) |
| hf | 예측시간 | 시간(hour)(dP:24=+24H) |
| disp | 제공형태 | A(ASCII) |

#### 3.2.1 단면도

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_xz_txt1?group=KIML&nwp=l010&data=P&varn=3005&lvl_lst=&tmfc=2026030100&hf=24&lon1=127.7&lat1=39.7&lon2=133.6&lat2=44.7&disp=A&authKey={인증키입력}

#### 3.2.2 특정 고도의 단면값

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_xz_txt1?group=KIML&nwp=l010&data=P&varn=3005&lvl_lst=1000,500&tmfc=2026030100&hf=24&map=F&lon1=127.7&lat1=39.7&lon2=133.6&lat2=44.7&disp=A&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| group | 모델 | KIMR(지역), KIML(국지) |
| nwp | 종류 | r030(지역), l010(국지) |
| data | 자료종류 | P(등압면), U(단일면) |
| varn | 변수종류 | D*100000 + C*1000 + P
- D : GRIB.sec0.discipline : 자료의 종류
- C : GRIB.sec4.parameter category : 변수 그룹
- P : GRIB.sec4.parameter number : 변수 번호 |
| tmfc | 분석시간 | 연월일시(UTC) |
| hf | 예측시간 | 시간(hour)(dP:24=+24H) |
| lvl_lst | 고도 | 격자점 자료 조회의 경우에 level값이 있으면 해당 고도값만 표출, 없으면 각 고도의 값을 모두 표출
예) 850hPa -> lvl_lst=850으로 입력 |
| lon1 | 시작경도 | 가장 가까운 격자점 값을 사용 |
| lat1 | 시작위도 | 가장 가까운 격자점 값을 사용 |
| lon2 | 종료경도 | 가장 가까운 격자점 값을 사용 |
| lat2 | 종료위도 | 가장 가까운 격자점 값을 사용 |
| disp | 제공형태 | A(ASCII) |

#### 3.3.1 임의 격자점

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_pt_txt1?group=KIMR&nwp=r030&data=P&varn=0,3005&tmfc=2026010100&hf=24&X=400&Y=500&disp=A&help=1&authKey={인증키입력}

#### 3.3.2 임의 고도, 임의 위경도

https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_pt_txt1?group=KIML&nwp=l010&data=P&varn=0&tmfc=2026010100&hf=0&lon=125.5&lat=37.5&level=850&help=0&authKey={인증키입력}

#### 요청인자

| **인자명** | **의미** | **설명** |
| --- | --- | --- |
| group | 모델 | KIMR(지역), KIML(국지) |
| nwp | 종류 | r030(지역), l010(국지) |
| data | 자료종류 | P(등압면), U(단일면) |
| varn | 변수종류 | D*100000 + C*1000 + P
- D : GRIB.sec0.discipline : 자료의 종류
- C : GRIB.sec4.parameter category : 변수 그룹
- P : GRIB.sec4.parameter number : 변수 번호 |
| tmfc | 분석시간 | 연월일시(UTC) |
| hf | 예측시간 | 시간(hour)(dP:24=+24H) |
| level | 고도 | 격자점 자료 조회의 경우에 level값이 있으면 해당 고도값만 표출, 없으면 각 고도의 값을 모두 표출
예) 850hPa -> level=850으로 입력 |
| X | X축-격자점위치 | 1부터 시작 |
| Y | Y축-격자점위치 | 1부터 시작 |
| lon | 경도 | 가장 가까운 격자점 값을 사용 |
| lat | 위도 | 가장 가까운 격자점 값을 사용 |
| disp | 제공형태 | A(ASCII) |



이후에도 엄청 많은데 너가 직접 찾아서 보고 필요한 자료를 수집할 수 있나?