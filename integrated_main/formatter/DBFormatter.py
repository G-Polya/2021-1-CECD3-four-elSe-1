
from .Formatter import Formatter
from .dictformat import dictformat
from .MongoDBhandler import MongoDBhandler

class DBFormatter(Formatter):
    def __init__(self, host, port):
        self.handler = MongoDBhandler()

    def format(self,b, label, img_name, croppedImg, filename):
        data = dictformat(b,label,img_name,croppedImg)

        self.handler.insert_item_one(data,"search_pool")

