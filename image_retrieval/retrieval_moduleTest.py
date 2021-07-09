from ImageRetrievalClass import ImageRetrievalClass
from multiprocessing import freeze_support
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def main():
    retrievalClass = ImageRetrievalClass("IncepResNet",True,False)

    retrievalClass.buildModel()
    retrievalClass.transform()
    retrievalClass.train()
    E_train, E_test = retrievalClass.predict()

    retrievalClass.reconstuctionVisualize()

    E_train = E_train.reshape((4734,512,512,3))
    E_test = np.array(E_test)
    print("E_train.shape : ", E_train.shape )
    print("E_test.shape : ", E_test.shape)




if __name__ == "__main__":
    freeze_support()
    main()
