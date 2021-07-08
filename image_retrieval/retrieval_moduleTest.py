from ImageRetrievalClass import ImageRetrievalClass
from multiprocessing import freeze_support

def main():
    retrievalClass = ImageRetrievalClass("IncepResNet",True,False)

    retrievalClass.buildModel()
    retrievalClass.transform()
    retrievalClass.train()
    E_train, E_test = retrievalClass.predict()

    retrievalClass.reconstuctionVisualize()

    # E_train := DimensionalityReductor
    # 

    calculator = retrievalClass.similarityCalculator(E_train)
    retrievalClass.retrieval(E_test, calculator)

if __name__ == "__main__":
    freeze_support()
    main()
