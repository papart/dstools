from dstools.Dataset.Dataset import Dataset
from abc import ABCMeta, abstractmethod

class OperationInterface(metaclass=ABCMeta):
    def __init__(self, name='Operation'):
        self.name = name

    @abstractmethod
    def apply(self, ds: Dataset) -> Dataset:
        pass

    @abstractmethod
    def fit(self, ds: Dataset):
        pass

    def fit_apply(self, ds: Dataset) -> Dataset:
        self.fit(ds)
        return self.apply(ds)

    def _recreate_dataset(self, ds: Dataset, **kwargs) -> Dataset:
        """
        Create a new dataset on the basis of the given one.
        :param ds: Dataset, which performs as a basis for a new one.
        :param kwargs: dictionary {parameter:value}, which is used 
        to initialize for the resulting dataset.
        Supported parameters: df, index, features, targets, predictions.
        If a parameter is not provided, its value is taken from the given dataset ds.
        """
        for argname in ['df', 'index', 'features', 'targets', 'predictions']:
            if argname not in kwargs:
                kwargs[argname] = getattr(ds, argname)
        return Dataset(**kwargs)

    # TODO: implement __str__ and __repr__
