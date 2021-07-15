
# 팀원

- 김희수: 팀장, 유사이미지 검색 모듈 개발

- 송승민: 데이터베이스 설계 및 구축

- 박범수: 객체탐지 모듈 개발

- 전문수: auto labeling, annotation

# 개발환경

- OS : 우분투 18.04.5 LTS

- conda 가상환경 만들어서 conda 명령어로 순서대로 설치

- conda env create -f environment.txt -n "env-name"

- pip install opencv_contrib_python
- pip install keras-retinanet==0.5.1




# dataset

- [구글 드라이브 다운로드 링크](https://drive.google.com/drive/folders/1LUcWabcn_bu5u9iSkQN7LKuIzLStX832?usp=sharing)

- integrated_main의 original_test, original_train에 붙여넣기

# Configuration

```
└── integrated_main
    ├── original_test : 객체를 탐지하고 탐지한 객체와 유사한 이미지를 검색할 test input 이미지 디렉토리
    ├── original_train : 유사 객체 DB를 생성하기 위한 이미지 디렉토리. COCO dataset에서 person데이터 1000개를 임의로 가져옴
    ├── detected_data
         ├── detected_from_test : 새로운이미지(original_test)에서 탐지된 객체들
         └── detect_from_train :  기존이미지(original_train)에서 탐지한 객체들로 만든 검색대상 객체DB

    ├── retrieval_output : 유사이미지 검색 결과를 저장하는 디렉토리
    ├── image_retrieval.py : 유사이미지 검색 모듈
    ├── object_detection.py : 객체탐지 모듈
    ├── makeDB.py : 검색 대상이 될 유사 객체 DB 생성
    └── detect_and_retrieval.py : test input 이미지에서 객체를 탐지하고 유사 객체를 검색

```

# 주요 모듈 설명
1. image_retrieval.py : 이미지 검색을 담당하는 모듈
2. object_detection.py : 객체탐지를 수행하는 모듈. 
3. makeDB.py : 기존 이미지(original_train)에서 객체를 탐지하여 검색의 대상이 되는 객체들의 DB(현재로썬 file DB로 위치는 detected_from_train)를 만든다
4. detect_and_retrieval.py : 새로운 이미지(original_test의 이미지)에서 객체를 탐지하고 탐지한 객체와 유사한 객체를 DB(detected_from_train)에서 찾아 결과를 리턴한다

# 실행순서

1. integrated_main 폴더에서 실행

2. python makeDB.py. 결과는 기존 이미지(original_train)에서 탐지한 객체들 이미지. detected_from_train에 저장된다

3. python detected_and_retrieval.py. original_test의 이미지에서 객체를 탐지하고 그 탐지한 객체와 유사한 객체를 detected_from_train에서 찾아 도출한다


# 오류목록
1. *.py나 주피터노트북에서 실행시켰는데 특정폴더(detected_from_train 등)에 권한이 없다(Permission denied)   
    >> chmod a+x,a+o,a+r "folder name" 로 모든 사용자에게 권한을 주면 해결됨. 권한이 없다면 root사용자로 들어가서 권한 부여