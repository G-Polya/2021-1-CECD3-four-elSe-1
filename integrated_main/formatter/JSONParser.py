import json
import numpy as np

def jsonParser(jsonFile):
    with open(jsonFile, "r") as inFile:
        obj = json.load(inFile)

    croppedImage = np.asarray(obj["croppedImage"])

    return obj["objectID"], obj["location"], obj["tag"], croppedImage, obj["IMG_URL"]

