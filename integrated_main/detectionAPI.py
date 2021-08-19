


from flask import Flask, render_template, request,jsonify
from flask_restful import Resource, Api
import tensorflow_hub as hub
from werkzeug.utils import secure_filename

import tensorflow as tf

from yolov5 import detect
import os



app = Flask(__name__)
api = Api(app)

# 업로드 HTML 렌더링
@app.route("/upload")
def render_file():
    return render_template("upload.html")


# 파일 업로드 처리
@app.route("/fileUpload", methods=["GET", "POST"])
def upload_file():
    if request.method =="POST":
        f = request.files["file"]
        filename = "./yolov5/hanssem/images/query/" + secure_filename(f.filename)
        f.save(filename)
        return "original_test 디렉터리 -> 파일 업로드 성공!"




class Detector(Resource):
    detector = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            # print("__new__ is called")
            cls._instance = super(Detector,cls).__new__(cls)
        return cls._instance

    def __init__(cls,module_handle="https://tfhub.dev/tensorflow/efficientdet/d0/1"):
        if cls.detector != None:
            print("Already exist")
        else:
            cls.detector =  hub.load(module_handle)

    def get(self):
        pass

    def getDetector(cls):
        return cls.detector



@app.route("/api")
class EfficientDet(Resource):

    def get(self):
        model =  Detector().getDetector()
        
        # dataset_path = "./original_train_person/"
        dataset_path = "./original_test/"
        output_path = "./detected_data/detected_from_test/"
        dataset_list = os.listdir(dataset_path)
        detected_objectList = object_detection(model, dataset_list,dataset_path, output_path) # detection에 약 8초
       
        return jsonify({"detected_objectList" : detected_objectList, "model":str(model)})

api.add_resource(EfficientDet, "/api/efficientDet")
api.add_resource(Detector, "/")


if __name__ =="__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    app.run()
    