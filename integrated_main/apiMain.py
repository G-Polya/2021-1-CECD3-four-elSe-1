from ImageRetrievalClass import ImageRetrievalClass
from flask_restful import Resource, Api
from detectionAPI import ModelLoader, Detection
from flask import Flask, render_template, request, jsonify
import time
import faiss


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
    if request.method == "POST":
        f = request.files["file"]

        filename = "./yolov5/hanssem/images/query/" + \
            secure_filename(f.filename)
        f.save(filename)
        return "/yolov5/hanssem/images/query/ 디렉터리 -> 파일 업로드 성공!"


class Delete(Resource):
    def get(self):
        print("current directory : ", os.getcwd())

        os.system("rm ./yolov5/hanssem/images/query/*")
        os.system("rm ./yolov5/runs/detect/ -rf")
        os.system("rm ./static/img/*")

        print("current directory : ", os.getcwd())

        print("delete completed")

        return "delete completed"

from ModelLoader import ModelLoader
def retrieval(idx):
    before = time.time()

    # idx = int(request.args["idx"])
    select = Select.getInstance()
    selectObject = select.getDetected()[idx]

    selectObject_path = selectObject["objectImagePath"]
    selectObject_pil = Image.open(selectObject_path)
    modelLoader = ModelLoader.getInstance()
    retrievalInstance = modelLoader.getRetrieval()
    # retrievalInstance = ImageRetrievalClass("IncepResNet", True, False)
    
    # retrievalInstance.buildModel()
    retrievalInstance.readTestSet(selectObject_pil)
    
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

    queryed_jsonList = query.getQueryed_jsonList()

    d = E_train_flatten.shape[1]
    index = faiss.IndexFlatL2(d)
    print("index.is_trained : ", index.is_trained)

    index.add(E_train_flatten)
    print("index.ntotal : ", index.ntotal)

    print("Using FAISS Index")
    k = 5
    D, retrieval_indices = index.search(E_test_flatten,k)

    similar_json = []
    similar_json_url = []
    for i in range(len(retrieval_indices[0])):
        temp = retrieval_indices[0][i]
        # print(temp)
        similar_json.append(queryed_jsonList[temp])
        similar_json_url.append(queryed_jsonList[temp]['IMG_URL'])

    after = time.time()
    elapsed_time = after - before
    print("Retrieval completed! " + str(round(elapsed_time, 2)) + "s")
    output = {
        "selectedObject": selectObject,
        "retrieval_output": similar_json
    }
    return output


@app.route("/api/retrieval/<int:idx>/showImage", methods=["GET", "POST"])
def showImage(idx):
    output = retrieval(idx)
    selectedObject_imagePath = output["selectedObject"]["objectImagePath"]
    os.system(f"cp ./{selectedObject_imagePath} ./static/img")

    selectedObject_imagePath = selectedObject_imagePath.split("/")[-1]
    imgURL_list = list()
    for retrieval_output in output["retrieval_output"]:
        imgURL_list.append(retrieval_output["IMG_URL"])

    return render_template("show.html", img_file="img/"+selectedObject_imagePath, urlList=imgURL_list)


@app.route("/api/retrieval/<int:idx>/getJSON", methods=["GET", "POST"])
def showJSON(idx):
    output = retrieval(idx)
    return output


api.add_resource(ModelLoader, "/")
api.add_resource(Detection, "/api/detection")
api.add_resource(Select, "/api/")
# api.add_resource(Retrieval, "/api/retrieval")
api.add_resource(Delete, "/delete")

if __name__ == "__main__":
    app.run(port=5050, debug=True)
