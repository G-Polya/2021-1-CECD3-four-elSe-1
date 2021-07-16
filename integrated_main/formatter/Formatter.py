
from abc import *


class Formatter(metaclass=ABCMeta):
    
    def __init__(self):
        pass

    @abstractmethod
    def format(b, label, img_name, croppedImg, filename):
        pass



