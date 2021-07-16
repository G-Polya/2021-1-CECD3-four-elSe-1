import json
import uuid
from .dictformat import dictformat

from .Formatter import Formatter

class JSONFormatter(Formatter):
    def __init__(self):
        pass

    def format(b, label, img_name, croppedImg, filename):
        detected = dictformat(b,label,img_name,croppedImg)

        with open(filename,"w") as outFile:
            json.dump(detected, outFile, indent=4)




