
from .SimpleAE import SimpleAE
from .StackedAE import StackedAE
class RetrievalModelFactory:
    def makeRetrievalModel(self,modelName,info):
        if modelName == "simpleAE":
            return SimpleAE(info)
        elif modelName == "stackedAE":
            return StackedAE(info)
            

        