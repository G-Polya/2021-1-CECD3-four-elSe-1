from yolov5.utils.torch_utils import select_device
from yolov5 import detect
from yolov5.utils.torch_utils import select_device, load_classifier, time_sync
from yolov5.models.experimental import attempt_load
import time
from flask_restful import Resource, Api

from ImageRetrievalClass import ImageRetrievalClass
class ModelLoader(Resource):
    _instance = None
    def __init__(self):
        if not ModelLoader._instance:
            print("ModelLoader.__init__ method called but nothing is created")
        else:
            print("ModelLoader instance already created", self.getInstance())

        before = time.time()
        self.__device = select_device('0')
        # weights='yolov5/runs/train/exp/weights/best.pt'
        weights = 'yolov5/runs/train/exp/weights/best.pt'
        self.__model =attempt_load(weights, map_location=self.__device) 
        self.__modelc = load_classifier(name="resnet50", n=2)

        self.__retrival = ImageRetrievalClass("IncepResNet",True,False)
        self.__retrival.buildModel(shape_img=(256,256,3))


        after = time.time()
        self.__elapsed_time = after-before

    def getModel(self):
        return self.__model

    def getRetrieval(self):
        return self.__retrival

    def getModelc(self):
        return self.__modelc

    def getDevice(self):
        return self.__device

    def getElapsed(self):
        return self.__elapsed_time    

    def get(self):
        # print(self.getInstance())
        print("Model Loaded! " + str(round(self.getElapsed(), 2))+"s")
        return "Model Loaded! " + str(round(self.getElapsed(), 2))+"s"

    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = ModelLoader()
        
        return cls._instance

    
