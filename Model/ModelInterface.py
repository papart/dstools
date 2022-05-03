import pandas as pd
from dstools.Dataset import Dataset
from abc import ABCMeta, abstractmethod

class ModelInterface(metaclass=ABCMeta):
    def __init__(self, name='Model'):
        self.name = name

    @abstractmethod
    def fit(self, ds: Dataset) -> 'ModelInterface':
        pass

    @abstractmethod
    def predict(self, ds: Dataset) -> Dataset:
        pass

    @abstractmethod
    def init_new(self, feats, hyperparams) -> 'ModelInterface':
        pass

    @property
    @abstractmethod
    def feats(self) -> list:
        pass
    
    @property
    @abstractmethod
    def targets(self) -> list:
        pass

    @property
    @abstractmethod
    def predicts(self) -> list:
        pass

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        pass        

    # hyperparams
    # base_model_class
    # get_base_model()
    # feats
    # targets
    # predicts?
    # TODO: properties for feats, targets, predicts
        # df = df[self.feats + self.targets]
        # return self._actual_fit(df)
    # reinit()

