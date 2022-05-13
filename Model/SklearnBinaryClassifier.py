import pandas as pd
from .SklearnMulticlassClassifier import SklearnMulticlassClassifier
from dstools.Dataset import Dataset

class SklearnBinaryClassifier(SklearnMulticlassClassifier):
    def __init__(self, features, target, prediction, hyperparams,
            core_model_class=None, core_model=None, name='SklearnBinaryClassifier'):
        # Assume that class_labels are 0, 1
        # prediction is a column for predicted probability of class 1
        super().__init__(features, target, [prediction], hyperparams, 
            core_model_class=core_model_class, core_model=core_model, name=name)

    def fit(self, ds: Dataset):
        unexpected_labels = set(ds.df[self.target]) - set(self.class_labels)
        if len(unexpected_labels) > 0:
            raise ValueError(f"Got unexpected labels: {unexpected_labels}")
        return super().fit(ds)

    def _core_model_predict(self, X: pd.DataFrame):
        return self._core_model.predict_proba(X)[:, 1]

    @property
    def prediction(self):
        return self._predicts[0]        

    @property
    def class_labels(self):
        return [0, 1]

    # TODO: docstrings
