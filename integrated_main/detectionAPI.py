from flask_restful import Resource, Api

from flask import jsonify
import os

from yolov5.utils.torch_utils import select_device
from yolov5 import detect
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.experimental import attempt_load
import time

class ModelLoader(Resource):
    model = None
    modelc = None
    device = None
    elapsed_time = None
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            # print("__new__ is called")
            cls._instance = super(ModelLoader,cls).__new__(cls)
        return cls._instance

    def __init__(cls):
        if cls.model != None or cls.modelc != None:
            print("Already exist")
        else:
            before = time.time()
            cls.device = select_device('0')
            weights='yolov5/runs/train/exp/weights/best.pt'
            cls.model = attempt_load(weights, map_location=cls.device)  
            cls.modelc = load_classifier(name="resnet50", n=2)
            after = time.time()
            cls.elapsed_time = after - before

    def get(self):
        print("Model Loaded! " + str(round(self.getElapsed(), 2))+"s")
        return "Model Loaded! " + str(round(self.getElapsed(), 2))+"s"

    def getElapsed(cls):
        
        return cls.elapsed_time

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
                                                      device=device, model=model, modelc=modelc,
                                                      iou_thres=0.1)

        return jsonify({"detected_objectList" : detectedObject_list})

