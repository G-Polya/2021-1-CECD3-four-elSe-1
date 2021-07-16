# 0. GCP VM 인스턴스 접속
1. 브라우저 SSH로 접속(권장)
2. 윈도우 [putty](https://newpower.tistory.com/192), [ssh](https://mentha2.tistory.com/210)
3. google cloud SDK로 접속 : gloud compute ssh "인스턴스 이름" [참고](https://shwksl101.github.io/gcp/2018/12/23/gcp_vm_custom_setting.html)
# 1. NVIDIA 그래픽 드라이버 설치
## 1) 기존 Nouveau 드라이버 제거 [참고](https://jhgan00.github.io/tip/2020/02/17/GCP_GPU/)
- GPU를 실제로 연산에 동원하기 위해서 NVIDIA 그래픽 드라이버를 설치해주어야 한다. 
- 우선 VM에 접속한 후 기존에 설치된 Nouveau드라이버를 제거
```
sudo apt-get remove nvidia* && sudo apt autoremove
sudo apt-get install dkms linux-headers-generic
sudo apt-get install build-essential linux-headers-`uname -r`
```
- /etc/modprobe.d/blacklist.conf 파일을 열어 내용을 수정
```
sudo nano /etc/modprobe.d/blacklist.conf

# 아래 내용 추가
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off

# 커널 재빌드
sudo update-initramfs -u
```

## 2) 그래픽 드라이버 설치 [참고](https://velog.io/@cychoi74/%EC%9A%B0%EB%B6%84%ED%88%AC-18.04-NVIDIA-%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B2%84-%EC%84%A4%EC%B9%98)
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update


ubuntu-drivers devices # 설치가능한 드라이버 목록 보여줌 

sudo apt-get install nvidia-driver-460(설치하려는 드라이버)
sudo reboot

nvidia-settings
nvidia-smi
```
# 1. CUDA 설치 [참고](https://choice-life.tistory.com/69)
## 1) Nvidia 홈페이지에서 CUDA 다운로드
- [홈페이지](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local)에서 linux > x86_64 > Ubuntu > 18.04 > deb(local), deb(network), runfile 중 하나로 설치. 여기선 deb(local)로 설명
## 2) 설치후 CUDA PATH 설정
- 현재 로그인 유저이름이 user라고 가정
```
user@instance-1:~$ vim .bashrc (home/user 디렉토리에 존재하는 .bashrc파일)

# 다음을 .bashrc에 추가
export CUDA_HOME = /usr/local/cuda
export LD_LIBRARY_PATH=${CUDA_HOME}/lib6
PATH = ${CUDA_HOME}/bin:${PATH}
export PATH
```

# 1. cuDNN 설치
## 1) cuDNN Library for Linux(x86_64)를 VM인스턴스에 업로드
- 방법1) 브라우저 SSH로 VM 인스턴스로 업로드(느리다)
- 방법2) google cloud sdk shell에서 gcloud 명령줄 도구를 이용해서 업로드
```
# gcloud compute scp "업로드하려는 파일의 경로" "VM인스턴스이름":"인스턴스안에서의 파일이름"
gcloud compute scp cudnn-11.3-linux-x64-v8.2.1.32.tgz instance-1:cudnn.tgz
```
## 2)  cuda 폴더 붙여넣기
- cudnn.tgz압축을 풀면 cuda폴더가 추출됨. 이것을 복사 붙여넣기
- 복사 붙여넣기 전. /usr/local 디렉토리에 cuda-xx.x폴더가 존재하는지 확인(앞서 CUDA 11.4를 설치했으니 cuda-11.4 폴더가 존재해야함)
```
tar -zxvf cudnn.tgz(압축폴더이름)
sudo cp cuda/include/cudnn.h /usr/local/cuda-xx.x/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-xx.x/lib64
```
## failed to initialize NVML: Driver / library version mismatch Error
[해결방법](https://jangjy.tistory.com/300)


# 4. 아나콘다 및 주피터 노트북 설치
## 1) 아나콘다 설치 
- 아나콘다 설치까지는 [링크](https://shwksl101.github.io/gcp/2018/12/23/gcp_vm_custom_setting.html)를 따른다
- 그후 environment.yml로 가상환경을 생성하는데 문제가 생긴다면 다음을 따른다
```
conda install --channel defaults conda python=3.7 --yes
conda update --channel defaults --all --yes
```
- 문제가 해결되었으면 environment.yml로 가상환경 설정
```
conda env create -f environment.yml -n "env_name"
```

## 2) 주피터 노트북 설정 [참고](https://john-analyst.medium.com/google-cloud-platform-vm%EC%97%90-jupyter-notebook-%EC%84%B8%ED%8C%85%ED%95%98%EA%B8%B0-135d226c4ffa)
- 방화벽 설정
- jupyter notebook --generate-config로 jupyter_notebook_config.py 생성
- vim ~/.jupyter/jupyter_notebook_config.py로 수정
```
c = get_config()
c.NoteBookApp.ip = "VM인스턴스의 외부 IP"
c.NoteBookApp.port = 8888
```
- 수정했으면 콘솔에서 jupyter notebook --ip=0.0.0.0 --port=8888로 주피터 노트북을 실행
- 브라우저 주소창에서 "VM인스턴스의 외부 IP":8888로 주피터노트북 접속
- 암호는 콘솔에서 나타난 token 

# 5. FileZilla로 파일업로드 및 다운로드
## 1) 윈도우
- 앞서 제시한 윈도우에서 putty를 활용하여 VM인스턴스에 접속한 것 참고 [링크](https://newpower.tistory.com/192)
- 이때 사용자명은 root로 설정하고 key passphrase를 설정할 것 
- public key는 VM 인스턴스의 SSH키에 저장해놓고, private key는 따로 보관
- [링크](https://artnfear.com/entry/GCP-%EA%B5%AC%EA%B8%80-%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C-%ED%94%8C%EB%9E%AB%ED%8F%BC%EC%97%90%EC%84%9C-SFTP-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)를 따라서 하면 private key 선택이 되지 않는 경우가 발생
- 이는 [링크](https://imflower.tistory.com/2808)를 참고해서 해결가능
- 또, 접속을 제대로 했다 하더라도 VM인스턴스에 파일질라를 이용해서 업로드, 다운로드, 디렉토리 생성이 안될텐데 이는 sudo 권한이 없기 때문
- 사용자이름이 root여야 하며 /etc/ssh/sshd_config에서 PermitRootLogin를 yes로 바꿔줘야 한다 [링크](https://www.siteyaar.com/connect-as-root-via-sftp-on-google-cloud/)