import pandas as pd
from .SklearnModel import SklearnModel
from dstools.Dataset import Dataset

class SklearnMulticlassClassifier(SklearnModel):
    def __init__(self, features, target, predictions, hyperparams,
            core_model_class=None, core_model=None, name='SklearnMulticlassClassifier'):
        # Assume that class_labels are 0, 1, ... len(predictions)-1
        super().__init__(features, [target], predictions, hyperparams, 
            core_model_class=core_model_class, core_model=core_model, name=name)

    def fit(self, ds: Dataset):
        unexpected_labels = set(ds.df[self.target]) - set(self.class_labels)
        if len(unexpected_labels) > 0:
            raise ValueError(f"Got unexpected labels: {unexpected_labels}")
        return super().fit(ds)

    def _core_model_predict(self, X: pd.DataFrame):
        return self._core_model.predict_proba(X)

    @property
    def target(self):
        return self._targets[0]

    @property
    def class_labels(self):
        return list(range(len(self.predictions)))
