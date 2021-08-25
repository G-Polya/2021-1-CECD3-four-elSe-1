from flask_restful import Resource, Api
import requests
from flask import Flask, request, jsonify
from glob import glob 
from PIL import Image
import numpy as np
import json
import time 

class Select(Resource):
    def get(self):
        pass
    
    @staticmethod
    def getSelectObject(idx):
        # idx = int(request.args["idx"])
        url = "http://127.0.0.1:5050/api/detection"
        response = requests.get(url)
        detected_objectList = response.json()["detected_objectList"]
        selectObject = None
        try:
            selectObject = detected_objectList[idx]
            return selectObject
        except IndexError as e:
            return e




# 추후 mongoDB로 변경
class Query:
    def __init__(self, tag):
        self.__tag = tag
        self.__jsonFiles_path = glob("./jsonFiles/*.json")
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

        


        

        
        

