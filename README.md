 # PPE 탐지 및 이상행동 탐지 프로젝트

---

## 팀원 초기 셋업 가이드

### STEP 1. Python 설치 확인
1. VSCode 열고 상단 메뉴 → Terminal → New Terminal 클릭

2. 터미널에 입력: python --version

Python 3.x.x 이라고 뜨면 설치된 거야.
안 뜨면 https://www.python.org/downloads/ 에서 설치
(설치할 때 Add Python to PATH 반드시 체크!)

---

### STEP 2. 레포 클론
1. 바탕화면에 새 폴더 만들기 (이름은 본인이 원하는 대로)

2. 폴더 경로 복사해서 터미널에 경로 복붙:
cd 본인 폴더 경로 (예시: cd C:\Users\본인이름\OneDrive\바탕 화면\폴더 이름)

3. 레포 클론하기:
git clone https://github.com/팀이름/레포이름.git

4. cd prestyle-project

---

### STEP 3. VSCode에서 폴더 열기
1. VSCode 상단 메뉴 → File → Open Folder

2. 방금 클론한 레포 폴더 선택 → 폴더 선택 클릭

3. 터미널 경로가 레포 폴더로 바뀌면 성공!

---

### STEP 4. 가상환경 만들기
1. 터미널에 입력:
python -m venv venv

2. 왼쪽 파일 탐색기에 venv 폴더가 생기면 성공!

---

### STEP 5. 가상환경 켜기
1. 터미널에 입력:
venv\Scripts\activate

2. 터미널 앞에 (venv) 표시가 뜨면 성공!
: (venv) C:\Users\... <- 이런식으로

3. VSCode 열 때마다 이 명령어 입력해야 해! (VSCode 열 때 떠있으면 굳이 안해도 됨)

---

### STEP 6. 패키지 설치
1. 터미널에 입력:
pip install -r requirements.txt

2. 설치가 쭉 진행되면 성공! (처음 한 번만 하면 돼, requirements가 바뀌면 또 해야되는데 바뀌면 내가 공지해줄게)

---

### STEP 7. 각자 본인 브랜치로 이동
1. 터미널에 입력:
git checkout feature/본인브랜치이름

2. 브랜치 목록:
- feature/model-dev : 모델 개발 담당 (창우, 다운)
- feature/data-pipeline : 데이터 전처리 담당 (동진, 정식)
- feature/webcam-test : 테스트 담당 (창우, 다운)
- feature/visualization : 시각화 담당 (미정)

3. 예시 (데이터 담당이라면):
git checkout feature/data-pipeline

4. 터미널에 Switched to branch 'feature/...' 라고 뜨면 성공!

---

## 매일 작업 시작 전 루틴 (필수!)

1. dev 브랜치 최신 코드 받기:

   - git checkout dev

   - git pull origin dev

2. 본인 브랜치로 이동:

   - git checkout feature/본인브랜치이름

3. dev 브랜치 최신 내용 본인 브랜치에 반영:

   - git merge dev

4. 작업 시작!

---

## 작업 끝나면

1. 변경된 파일 확인:

   - git status

2. 전체 파일 올릴 준비 (add):

   - git add . ("add" + "띄어쓰기" + "." 해야됨)

3. 커밋 (commit) (어떤 작업 했는지 메시지 남기기):

   - git commit -m "작업 내용 간단히 설명"

4. GitHub에 올리기 (push):

   - git push origin feature/본인브랜치이름

5. GitHub 레포 페이지에서 Pull Request 생성

    → base는 dev, compare는 feature/본인브랜치이름 으로 설정

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
