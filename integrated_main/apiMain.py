from flask_restful import Resource, Api
from detectionAPI import ModelLoader, Detection
from flask import Flask, render_template, request,jsonify
import time

from retrievalAPI import Select, Query
from PIL import Image
import numpy as np

from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
api = Api(app)

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
        return "/yolov5/hanssem/images/query/ 디렉터리 -> 파일 업로드 성공!"

class Delete(Resource):
    def get(self):
        os.chdir("./yolov5/hanssem/images/query/")
        print("current directory : ",os.getcwd())
        os.system("rm *.jpg")

        os.chdir("../../../runs")
        print("current directory : ",os.getcwd())
        os.system("rm -rf detect/")

        os.chdir("../../../")
        print("current directory : ",os.getcwd())
        
        print("delete completed")
        return "delete completed"


 
from ImageRetrievalClass import ImageRetrievalClass 

def retrieval(idx):
    before = time.time()

    # idx = int(request.args["idx"])
    selectObject = Select.getSelectObject(idx)
    
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
    output = {
        "selectedObject": selectObject,
        "retrieval_output":similar_json
    }
    return output       

@app.route("/api/retrieval/<int:idx>/show", methods=["GET","POST"])
def show(idx):
    print(idx)
    output = retrieval(idx)
    # print(output)
    return render_template("show.html", output=output)
    


api.add_resource(ModelLoader,"/")
api.add_resource(Detection,"/api/detection")
api.add_resource(Select, "/api/select")
# api.add_resource(Retrieval, "/api/retrieval")
api.add_resource(Delete,"/delete")

if __name__ == "__main__":
    app.run(port=5050, debug=True)