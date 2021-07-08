
import keras
from .SimpleAE import SimpleAE
from .StackedAE import StackedAE
from .Resnet50AE import Resnet50AE
from .DELF import DELF

class RetrievalModelFactory:
    def makeRetrievalModel(self,modelName,info):
        if modelName == "simpleAE":
            return SimpleAE(info)
        elif modelName == "Resnet50AE":
            return Resnet50AE(info)
        elif modelName == "stackedAE":
            return StackedAE(info)
        elif modelName == "DELF":
            return DELF(info)
            

        