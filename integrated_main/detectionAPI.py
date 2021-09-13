from flask_restful import Resource, Api
from ImageRetrievalClass import ImageRetrievalClass
from flask import jsonify
import os

from yolov5.utils.torch_utils import select_device
from yolov5 import detect
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.experimental import attempt_load
import time
from ModelLoader import ModelLoader

    

class Detection(Resource):

    def get(self):
        modelLoader = ModelLoader.getInstance()
        device = modelLoader.getDevice()
        model =  modelLoader.getModel()
        modelc = modelLoader.getModelc()
        
        detectedObject_list = detect.object_detection(imgsz=[576],name="query",
                                                      source="yolov5/hanssem/images/query",
                                                      device=device, model=model, modelc=modelc,
                                                      iou_thres=0.1)

        return jsonify({"detected_objectList" : detectedObject_list})

