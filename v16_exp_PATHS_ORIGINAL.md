# 팀원 원본 코드 경로 정규화 기록 (부록 T 패키징, 2026-07-27)

## 배경
`v7_cnn_submission.py`·`spatial_cnn_v7_top.py`는 팀원 원본 코드를 감사·재현 목적으로 우리 저장소에
그대로 들여온 사본이다. 두 파일 모두 팀원 계정(`/home/2022113196`)의 절대경로를 하드코딩하고 있었고,
그중 데이터 경로는 한글(`풍력발전/풍력발전/`)을 포함해 NFD/NFC 인코딩 사고(2026-07-25 발견·전체
정규화 이력, `DACON_풍력_전체기록.md` §11 참조)를 재발시킬 위험이 있었다.

## 원본 하드코딩 경로 (수정 전, 원문 그대로 보존)
```
BASE      = "/home/2022113196/v16_exp"
DATA      = "/home/2022113196/풍력발전/풍력발전/dataset/train"
TEST_DATA = "/home/2022113196/풍력발전/풍력발전/dataset/test"   # v7_cnn_submission.py만 해당
```

## 정규화 후 (ASCII 상대경로)
```python
BASE = os.path.dirname(os.path.abspath(__file__))   # 우리 프로젝트 루트
DATA = os.path.join(BASE, "data")
TEST_DATA = os.path.join(BASE, "data")
```

## 근거 — 동일 데이터가 이미 우리 프로젝트에 존재함을 확인
`BASE`가 가리키던 파일(`merged_train_v9.csv`, `merged_test_v9.csv`, `cleaned_train_v3.csv`,
`submission_ficr_w1_v7.csv`)은 프로젝트 루트에 이미 동일하게 존재(팀 공유 파일). `DATA`/`TEST_DATA`가
가리키던 격자 원자료(`ldaps_train.csv`, `gfs_train.csv`, `ldaps_test.csv`, `gfs_test.csv`)도
`data/` 폴더에 이미 전부 존재(2026-07-16 원본 배포분, 팀 공통). 즉 **팀원 계정 접근 없이도 두 스크립트가
그대로 실행 가능** — 경로만 우리 환경의 동일 파일을 가리키도록 교체했을 뿐, 로직·아키텍처·seed는
한 글자도 바꾸지 않았다.

## 미해결 — `/home/2022113196/v16_exp` 자체 접근 불가
```
$ ls -la /home/2022113196/
ls: cannot open directory '/home/2022113196/': Permission denied
$ stat -c '%a %U %G' /home/2022113196
700 2022113196 domain users
```
이 두 파일(`v7_cnn_submission.py`, `spatial_cnn_v7_top.py`) 외에 `v16_exp` 폴더에 우리가 아직 사본을
확보하지 못한 추가 코드(예: 팀원의 다른 실험 스크립트, 체크포인트, 로그)가 있을 가능성은 배제 못함 —
디렉토리 자체가 소유자 전용(700) 권한이라 클로드코드(서버 실행 계정)로는 목록조회조차 불가.
**팀원이 직접 파일을 복사해 우리 저장소로 옮기거나, 해당 디렉토리에 읽기 권한(`chmod o+rx` 또는
그룹 공유)을 부여해야 확인·패키징 가능** — 이 부분은 사용자/팀원 조치 대기.
