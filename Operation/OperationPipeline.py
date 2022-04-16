from dstools.Dataset.Dataset import Dataset
from dstools.Operation.OperationInterface import OperationInterface

class OperationPipeline(OperationInterface):
    def __init__(self, stages, name='OperationPipeline'):
        self.stages = stages
        super().__init__(name=name)

    def apply(self, ds: Dataset) -> Dataset:
        for op in self.stages:
            ds = op.apply(ds)
        return ds

    def fit(self, ds: Dataset):
        for op in self.stages[:-1]:
            ds = op.fit_apply(ds)
        self.stages[-1].fit()

    def fit_apply(self, ds: Dataset) -> Dataset:
        for op in self.stages:
            ds = op.fit_apply(ds)
        return ds        

    @property
    def stages(self):
        return self._stages
    
    @stages.setter
    def stages(self, new_stages):
        if not isinstance(new_stages, list):
            raise TypeError("OperationPipeline.stages must be a list")
        if not all([isinstance(st, OperationInterface) for st in new_stages]):
            raise TypeError("All stages must be instances of OperationInterface")
        self._stages = new_stages

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("OperationPipeline's keys must be integers")
        if key < 0 or key >= len(self.stages):
            raise ValueError("Index out of range")
        return self.stages[key]

    def __setitem__(self, key, op):
        if not isinstance(op, OperationInterface):
            raise TypeError(
                "Every OperationPipeline's stage"
                " must be an instance of OperationInterface")
        self.stages[key] = op
    
    # TODO: implement len()
    # TODO: docstrings
