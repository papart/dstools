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
    def init_new(self, features, hyperparams) -> 'ModelInterface':
        pass

    @property
    @abstractmethod
    def features(self) -> list:
        pass
    
    @property
    @abstractmethod
    def targets(self) -> list:
        pass

    @property
    @abstractmethod
    def predictions(self) -> list:
        pass

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        pass        

    # hyperparams
    # base_model_class
    # get_base_model()
    # features
    # targets
    # predictions?
    # TODO: properties for features, targets, predictions
        # df = df[self.features + self.targets]
        # return self._actual_fit(df)
    # reinit()

