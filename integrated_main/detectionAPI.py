


from yolov5.utils.torch_utils import select_device
from flask import Flask, render_template, request,jsonify
from flask_restful import Resource, Api

from werkzeug.utils import secure_filename

from yolov5 import detect
import os

from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.experimental import attempt_load


app = Flask(__name__)
api = Api(app)

# 업로드 HTML 렌더링
@app.route("/upload", methods=["GET"])
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



@app.route("/api")
class Detector(Resource):
    model = None
    modelc = None
    device = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            # print("__new__ is called")
            cls._instance = super(Detector,cls).__new__(cls)
        return cls._instance

    def __init__(cls):
        if cls.model != None or cls.modelc != None:
            print("Already exist")
        else:
            cls.device = select_device('0')
            weights='yolov5/runs/train/exp/weights/best.pt'
            cls.model = attempt_load(weights, map_location=cls.device)  
            cls.modelc = load_classifier(name="resnet50", n=2)

    def get(self):
        pass

    def getSelf(self):
        return self

    def getModel(cls):
        return cls.model
    
    def getModelc(cls):
        return cls.modelc

    def getDevice(cls):
        return cls.device


# @app.route("/api/detection")
class detection(Resource):

    def get(self):
        detector = Detector()
        device = detector.getDevice()
        model =  detector.getModel()
        modelc = detector.getModelc()
        
        detectedObject_list = detect.object_detection(imgsz=[576],name="query",
                                                      source="yolov5/hanssem/images/query",
                                                      device=device, model=model, modelc=modelc)

        return jsonify({"detected_objectList" : detectedObject_list})


from ImageRetrievalClass import ImageRetrievalClass
def Retrieval(Resource):
    def get(self):
        url = "http://127.0.0.1:5000/api/detection"
        response = requests.get(url)
        response.json()
        print(response.json())



api.add_resource(detection, "/api/detection")
api.add_resource(Retrieval, "/api/retrieval")
api.add_resource(Detector, "/")


if __name__ =="__main__":
    app.run()
    