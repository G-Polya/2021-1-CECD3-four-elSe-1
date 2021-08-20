from flask_restful import Resource, Api

from flask import jsonify
import os

from yolov5.utils.torch_utils import select_device
from yolov5 import detect
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.experimental import attempt_load

class ModelLoader(Resource):
    model = None
    modelc = None
    device = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            # print("__new__ is called")
            cls._instance = super(ModelLoader,cls).__new__(cls)
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


class Detection(Resource):

    def get(self):
        modelLoader = ModelLoader()
        device = modelLoader.getDevice()
        model =  modelLoader.getModel()
        modelc = modelLoader.getModelc()
        
        detectedObject_list = detect.object_detection(imgsz=[576],name="query",
                                                      source="yolov5/hanssem/images/query",
                                                      device=device, model=model, modelc=modelc)

        return jsonify({"detected_objectList" : detectedObject_list})

