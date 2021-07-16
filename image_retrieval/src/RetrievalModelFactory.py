
import keras
from .SimpleAE import SimpleAE
from .StackedAE import StackedAE
from .VggAE import VggAE
from .DELF import DELF

class RetrievalModelFactory:
    def makeRetrievalModel(self,modelName,info):
        if modelName == "simpleAE":
            return SimpleAE(info)
        elif modelName == "vggAE":
            return VggAE(info)
        elif modelName == "stackedAE":
            return StackedAE(info)
        elif modelName == "DELF":
            return DELF(info)
            

        