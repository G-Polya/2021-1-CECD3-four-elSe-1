# ### keras-retinanet 패키지를 이용하여 이미지와 영상 Object Detection 수행
# *  Pretrained된 coco 모델을 로드 하고 이를 이용하여 Object Detection 수행

# #### 관련 모듈 import 
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import base64
import codecs, json
import pickle
import matplotlib.pyplot as plt
import keras
from PIL import Image
import uuid
# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#from keras_retinanet.utils.gpu import setup_gpu
from keras.models import Model



# #### keras-retinanet으로 pretrained된 coco 모델 다운로드하고 해당 모델을 로드
# 아래 모델은 https://github.com/fizyr/keras-retinanet/releases 에서 download 받을 수 있음. 
# 해당 모델 h5 파일을 snapshot 디렉토리에 저장 후 retina model의 load_model()을 이용하여 모델 로딩.
# !wget https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5 


# pretrained coco 모델 파일을 retinanet 모델로 로딩.  


# # Feature Map 출력

# #### coco 데이터 세트의 클래스id별 클래스명 지정. 
labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' 
                      }

labels_to_num = [0]*len(labels_to_names_seq)


# load image dataset


def object_detection(model, inputData_list, dataset_path,output_path):

    formatList = list()

    for img_name in inputData_list:
        print("handling "+img_name)
        imagePath = dataset_path + img_name
        image = read_image_bgr(imagePath)

        print('image shape:', image.shape)
        
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # 모델에 입력전에 이미지 사전 처리. keras-retinanet은 image
        image = preprocess_image(image)
        image, scale = resize_image(image)
        print('resized image size:', image.shape, 'scale:', scale)

        # 이미지에 대해 Object Detection 수행. 
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print(boxes.shape, scores.shape, labels.shape)
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            
            labels_to_num[label] += 1

            b = box.astype(int)
            #print(b)
            object_img = draw[b[1]:b[3],b[0]:b[2]]
            
            
            object_img = Image.fromarray(object_img)
            object_img = object_img.resize((512,512))

            imagePath_str = imagePath.replace('/','-')

            # 객체 dump
            # os.chdir(output_path)
            tag = labels_to_names_seq[label]
            filename = output_path + "{}_path: ({}).jpg".format(labels_to_names_seq[label]+str(labels_to_num[label]),imagePath_str)
            object_img.save(filename)
            format = {
                "objectID" : str(uuid.uuid4()),
                "location":{
                    "xmin":b[1].item(),
                    "ymin":b[3].item(),
                    "xmax":b[0].item(),
                    "ymax":b[2].item()
                },
                "tag": str(tag),
                "objectPath" : filename,
                "IMG_URL" : img_name
            }

            formatList.append(format)
            # os.chdir('../')
    print("detection 완료!")

    return formatList
  
