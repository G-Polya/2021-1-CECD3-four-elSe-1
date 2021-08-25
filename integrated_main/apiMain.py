from flask_restful import Resource, Api
from detectionAPI import ModelLoader, Detection
from retrievalAPI import Select, Retrieval
from flask import Flask, render_template, request,jsonify


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
        


api.add_resource(ModelLoader,"/")
api.add_resource(Detection,"/api/detection")
api.add_resource(Select, "/api/select")
api.add_resource(Retrieval, "/api/retrieval")
api.add_resource(Delete,"/delete")

if __name__ == "__main__":
    app.run(port=5050, debug=True)