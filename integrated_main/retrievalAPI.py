from flask_restful import Resource, Api
import requests
from flask import Flask, request
from glob import glob 
from PIL import Image
import numpy as np
import json
import time 

class Select(Resource):
    def get(self):
        pass
    
    @staticmethod
    def getSelectObject():
        idx = int(request.args["idx"])
        url = "http://127.0.0.1:5000/api/detection"
        response = requests.get(url)
        detected_objectList = response.json()["detected_objectList"]
        selectObject = None
        try:
            selectObject = detected_objectList[idx]
            return selectObject
        except IndexError as e:
            return e

from ImageRetrievalClass import ImageRetrievalClass   
class Retrieval(Resource):
   
    def get(self):
        before = time.time()
        selectObject = Select.getSelectObject()
        
        selectObject_path = selectObject["objectImagePath"]
        selectObject_pil = Image.open(selectObject_path)
        retrievalInstance = ImageRetrievalClass("IncepResNet", True, False)
        retrievalInstance.readTestSet(selectObject_pil)
        retrievalInstance.buildModel()

        X_test = retrievalInstance.testTransform()

        E_test = retrievalInstance.predictTest(X_test)
        E_test_flatten = E_test.reshape((-1, np.prod(retrievalInstance.output_shape_model)))
        tag = selectObject["tag"]
        print("tag : ", tag)
        
        query = Query(tag)
        E_train = query.get_E_train()
        print("E_train.shape : ", E_train.shape)
        E_train_flatten = E_train.reshape((-1, np.prod(retrievalInstance.output_shape_model)))
        print("E_train_flatten.shape : ", E_train_flatten.shape)


        calculator = retrievalInstance.similarityCalculator(E_train_flatten)
        queryed_jsonList = query.getQueryed_jsonList()
        retrieval_imagePool = [Image.open(json["objectImagePath"]) for json in queryed_jsonList]
        retrieval_indices = retrievalInstance.retrieval(E_test_flatten,calculator,retrieval_imagePool)
        
        similar_json=[]
        similar_json_url=[]
        for i in range(5):
            temp=retrieval_indices[0][i]
            # print(temp)
            similar_json.append(queryed_jsonList[temp])
            similar_json_url.append(queryed_jsonList[temp]['IMG_URL'])
        
        after = time.time()
        elapsed_time = after - before 
        print("Retrieval completed! " + str(round(elapsed_time, 2)) +"s")
        return similar_json

        



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

        


        

        
        

