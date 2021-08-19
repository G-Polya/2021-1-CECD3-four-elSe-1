# Train
/integrated_main/yolov5에서 
- single GPU : python train.py --batch 64 --data hanssem.yaml --weights yolov5s.pt --device 0
- multi GPU : python -m torch.distributed.launch --nproc_per_node 2 train.py --batch 64 --data hanssem.yaml --weights yolov5s.pt --device 0,1

# 검색대상이 될 retrieval object pool 생성
- retrieval object pool은 검색을 수행하기 전에 미리 만들어놔야한다

# inference and retrieval
- inference.ipynb에서 실행

# 