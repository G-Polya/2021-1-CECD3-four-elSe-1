
# 팀원

- 김희수: 팀장, 유사이미지 검색 모듈 개발

- 송승민: 데이터베이스 설계 및 구축

- 박범수: 객체탐지 모듈 개발

- 전문수: auto labeling, annotation

# 개발환경

- OS : 우분투 18.04.5 LTS

- conda 가상환경 만들어서 conda 명령어로 순서대로 설치

- python == 3.6.5

- tensorflow-gpu == 1.15

- scikit-learn 최신

- scikit-image 최신

- cudatoolkit == 10.0.130(tensorflow-gpu 1.15설치시 자동 설치)

- cudnn == 7.6.5(tensorflow-gpu 1.15설치시 자동 설치)

- cv2(pip install opencv_contrib_python)

- cython

- keras == 2.3.0(pip로 설치)

- tqdm

- keras-retinanet(pip로 설치)

# dataset

- [구글 드라이브 다운로드 링크](https://drive.google.com/drive/folders/1LUcWabcn_bu5u9iSkQN7LKuIzLStX832?usp=sharing)

- integrated_main의 original_test, original_train에 붙여넣기

# Configuration

```
└── integrated_main
    ├── original_test : 객체를 탐지하고 탐지한 객체와 유사한 이미지를 검색할 test input 이미지 디렉토리
    ├── original_train : 유사 객체 DB를 생성하기 위한 train input 이미지 디렉토리
    ├── detected_data
         ├── detected_from_test : original_test에서 탐지한 객체들
         └── detect_from_train :  original_train에서 탐지한 유사객체 DB

    ├── retrieval_output : 유사이미지 검색 결과를 저장하는 디렉토리
    ├── image_retrieval.py : 유사이미지 검색 모듈
    ├── object_detection.py : 객체탐지 모듈
    ├── makeDB.py : 검색 대상이 될 유사 객체 DB 생성
    └── detect_and_retrieval.py : test input 이미지에서 객체를 탐지하고 유사 객체를 검색

```

# 실행순서

1. integrated_main 폴더에서 실행

2. python makeDB.py 혹은 makeDB.ipynb 순차적으로 실행

3. python detected_and_retrieval.py 혹은 detected_and_retrieval.ipynb 순차적으로 실행
