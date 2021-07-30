import tensorflow as tf
import time
import tensorflow_hub as hub
import cv2
from PIL import Image
import uuid

#1부터 91까지의 COCO Class id 매핑
labels_to_names = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
                    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
                    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
                    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
                    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
                    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
                    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
                    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
                    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
                    91:'hair brush'}

labels_to_num = [0]*len(labels_to_names)

def get_detector(module_handle="https://tfhub.dev/tensorflow/efficientdet/d0/1"):
    detector=hub.load(module_handle)
    return detector


def get_detected_img(model, inputData_list, dataset_path, output_path):
    score_threshold = 0.4
    object_show_count = 100
    formatList = list()
    draw_imgs = list()
    
    for img_name in inputData_list:
        imagePath = dataset_path + img_name
        
        
        img_array = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        height = img_array.shape[0]
        width = img_array.shape[1]

        draw_img = img_array.copy()
        # draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)

        green_color = (0,255,0)
        red_color = (0,0,255)

        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.uint8)[tf.newaxis, ...]

        start_time = time.time()

        result = model(img_tensor)

        result = { key: value.numpy() for key, value in result.items()}

        for i in range(min(result["detection_scores"][0].shape[0], object_show_count)):
            score = result["detection_scores"][0,i]
            if score < score_threshold:
                break
            
            imagePath_str = imagePath.replace("/","-")

            box = result["detection_boxes"][0,i]


            ''' **** 주의 ******
            box는 ymin, xmin, ymax, xmax 순서로 되어 있음. '''
            left = int(box[1] * width)
            top = int(box[0] * height)
            right = int(box[3] * width)
            bottom = int(box[2] * height)

            detected_img = draw_img[left:right, top:bottom]
            detected_img = Image.fromarray(detected_img).resize((512,512))
            
            label = result["detection_classes"][0,i]

            labels_to_num[label] += 1

            tag = labels_to_names[label]
            
            filename = output_path + "{}_path_({}).jpg".format(labels_to_names[label]+str(labels_to_num[label]),imagePath_str)
            # filename = output_path+"_{}.jpg".format(tag)
            detected_img.save(filename)

            format = {
                "objectID" : str(uuid.uuid4()),
                "location" : {
                    "xmin" : top,
                    "ymin" : left,
                    "xmax" : bottom,
                    "ymax" : right
                },
                "tag" : str(tag),
                "objectImagaPath" : filename,
                "IMG_URL":img_name
            }

            caption = "{}: {:.4f}".format(tag, score)
            print(caption)

            cv2.rectangle(draw_img, (left, top), (right, bottom), color=green_color, thickness=2)
            cv2.putText(draw_img, caption, (left, (top-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red_color,1)


            
            print('Detection 수행시간:',round(time.time() - start_time, 2),"초")
            
            formatList.append(format)
            draw_imgs.append(draw_img)
            
    return formatList, draw_imgs
