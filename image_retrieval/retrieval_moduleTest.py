from ImageRetrievalClass import ImageRetrievalClass
from multiprocessing import freeze_support
import numpy as np
from sklearn.decomposition import PCA

def main():
    retrievalClass = ImageRetrievalClass("IncepResNet",True,False)

    retrievalClass.buildModel()
    retrievalClass.transform()
    retrievalClass.train()
    E_train, E_test = retrievalClass.predict()

    retrievalClass.reconstuctionVisualize()

    print("E_train.shape : ", E_train.shape )
    print("E_test.shape : ", E_test.shape)

if __name__ == "__main__":
    freeze_support()
    main()
