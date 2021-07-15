import uuid

def dictformat(b, label, img_name, croppedImg):
    format = {

        "objectID": str(uuid.uuid4()),
        "location":{
            "xmin":b[1].item(),
            "ymin":b[3].item(),
            "xmax":b[0].item(),
            "ymax":b[2].item()
        },
        "tag": str(label),
        "croppedImage":croppedImg,
        "IMG_URL" : img_name
    }

    return format 