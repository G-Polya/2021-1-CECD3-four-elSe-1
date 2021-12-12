from flask_restful import Resource, Api
import requests
from flask import Flask, request, jsonify
from glob import glob 
from PIL import Image
import numpy as np
import json
import time 



class Select(Resource):
    _instance = None
    def get(self):
        return "object selected"

    def __init__(self):
        if not Select._instance:
            print("Select.__init__ method called but nothing is created")
        else:
            print("Select instance already created", self.getInstance())
        url = "http://127.0.0.1:5050/api/detection"
        response = requests.get(url)
        self.detected_objectList = response.json()["detected_objectList"]
        
    def getDetected(self):
        return self.detected_objectList


    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = Select()

        return cls._instance


# 추후 mongoDB로 변경
class Query:
    def __init__(self, tag):
        self.__tag = tag
        self.__jsonFiles_path = glob("./mobileJson/*.json")
        self.__queryed_jsonList = list()
    
    
    def setSameTag_list(self):
        # queryed_json_pathList = list()
        

        for path in self.__jsonFiles_path:
            with open(path, "rb") as f:
                jsonFile = json.load(f)

            if jsonFile["tag"] == self.__tag:
                # queryed_json_pathList.append(path)
                self.__queryed_jsonList.append(jsonFile)
        
    def getQueryed_jsonList(self):
        return self.__queryed_jsonList

    def get_E_train(self):
        E_train = list()
        self.setSameTag_list()
        
        for json in self.__queryed_jsonList:
            npy = np.load(json["npyPath"])
            E_train.append(npy)
        E_train = np.array(E_train)
        return E_train

        


        

        
        

