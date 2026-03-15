 # PPE 탐지 및 이상행동 탐지 프로젝트

---

## 팀원 초기 셋업 가이드

### STEP 1. Python 설치 확인
VSCode 열고 상단 메뉴 → Terminal → New Terminal 클릭
터미널에 입력:
python --version

Python 3.x.x 이라고 뜨면 설치된 거야.
안 뜨면 https://www.python.org/downloads/ 에서 설치
(설치할 때 Add Python to PATH 반드시 체크!)

---

### STEP 2. 레포 클론
터미널에서 원하는 위치로 이동 (바탕화면 추천):
cd C:\Users\본인이름\OneDrive\바탕 화면

새 폴더 만들기:
mkdir ppe_project
cd ppe_project

레포 클론하기:
git clone https://github.com/팀이름/레포이름.git
cd 레포이름

---

### STEP 3. VSCode에서 폴더 열기
VSCode 상단 메뉴 → File → Open Folder
방금 클론한 레포 폴더 선택 → 폴더 선택 클릭
터미널 경로가 레포 폴더로 바뀌면 성공!

---

### STEP 4. 가상환경 만들기
터미널에 입력:
python -m venv venv

왼쪽 파일 탐색기에 venv 폴더가 생기면 성공!

---

### STEP 5. 가상환경 켜기
터미널에 입력:
venv\Scripts\activate

터미널 앞에 (venv) 표시가 뜨면 성공!
(venv) C:\Users\...>

VSCode 열 때마다 이 명령어 입력해야 해!

---

### STEP 6. 패키지 설치
터미널에 입력:
pip install -r requirements.txt

설치가 쭉 진행되면 성공! (처음 한 번만 하면 돼)

---

### STEP 7. 내 브랜치로 이동
터미널에 입력:
git checkout feature/본인브랜치이름

브랜치 목록:
- feature/model-dev : 모델 개발 담당
- feature/data-pipeline : 데이터 전처리 담당
- feature/webcam-test : 테스트 담당
- feature/visualization : 시각화 담당

예시 (데이터 담당이라면):
git checkout feature/data-pipeline

터미널에 Switched to branch 'feature/...' 라고 뜨면 성공!

---

## 매일 작업 시작 전 루틴 (필수!)

1. dev 최신 코드 받기
git checkout dev
git pull origin dev

2. 내 브랜치로 이동
git checkout feature/내브랜치이름

3. dev 최신 내용 내 브랜치에 반영
git merge dev

4. 작업 시작!

---

## 작업 끝나면

1. 변경된 파일 확인
git status

2. 전체 파일 올릴 준비
git add .

3. 커밋 (어떤 작업 했는지 메시지 남기기)
git commit -m "작업 내용 간단히 설명"

4. GitHub에 올리기
git push origin feature/내브랜치이름

5. GitHub 레포 페이지에서 Pull Request 생성
→ base: dev, compare: feature/내브랜치이름 로 설정
→ Create pull request 클릭

---

## 브랜치 역할
- main : 최종 완성본 (직접 push 금지!)
- dev : 통합 테스트 공간
- feature/model-dev : 모델 개발
- feature/data-pipeline : 데이터 전처리
- feature/webcam-test : 테스트
- feature/visualization : 시각화

---

## 데이터 & 모델
- 데이터셋 : Google Drive (링크 추가 예정)
- 모델 가중치(.pt) : Google Drive (링크 추가 예정)