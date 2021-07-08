<<<<<<< HEAD

from image_retrieval.src.RetrievalModelFactory import RetrievalModelFactory
from .utils import split
from .AbstractRetrievalModel import AbstractRetrievalModel

class DelfRetrievalModel(AbstractRetrievalModel):
    def __init__(self,modelName, info):
        super(DelfRetrievalModel, self).__init__(modelName, info)

    def fit(self, X, n_epochs=50, batch_size=32, callbacks=None):
        indices_fracs = split(fracs=[0.9, 0.1], N=len(X), seed=0)
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        self.delf.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid),
                             callbacks=callbacks)

    def predict(self, X):
        return self.delf.predict(X)
    
    def set_arch(self):
        retrievalModelFactory = RetrievalModelFactory()
        self.delf = retrievalModelFactory.make
=======

from image_retrieval.src.RetrievalModelFactory import RetrievalModelFactory
from .utils import split
from .AbstractRetrievalModel import AbstractRetrievalModel

class DelfRetrievalModel(AbstractRetrievalModel):
    def __init__(self,modelName, info):
        super(DelfRetrievalModel, self).__init__(modelName, info)

    def fit(self, X, n_epochs=50, batch_size=32, callbacks=None):
        indices_fracs = split(fracs=[0.9, 0.1], N=len(X), seed=0)
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        self.delf.fit(X_train, X_train,
                             epochs=n_epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(X_valid, X_valid),
                             callbacks=callbacks)

    def predict(self, X):
        return self.delf.predict(X)
    
    def set_arch(self):
        retrievalModelFactory = RetrievalModelFactory()
        self.delf = retrievalModelFactory.make
>>>>>>> f72ad1f9408fc1fe4d43d73d29c241a660890638
