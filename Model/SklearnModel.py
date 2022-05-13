import copy
import pandas as pd
from dstools.Dataset import Dataset
from .ModelInterface import ModelInterface

class SklearnModel(ModelInterface):
    def __init__(self, feats, targets, predicts, hyperparams, 
            core_model_class=None, core_model=None, name='SklearnModel'):
        self.feats = feats
        self.targets = targets
        self.predicts = predicts
        self.hyperparams = hyperparams
        self._set_core_model(core_model_class, core_model)
        super().__init__(name=name)

    def fit(self, ds: Dataset):
        self._check_columns_in_ds(ds, 'feats')
        self._check_columns_in_ds(ds, 'targets')
        df = ds.df[ds.feats + ds.targets]
        X = df[ds.feats][self.feats]
        Y = df[ds.targets][self.targets]
        self._core_model.fit(X, Y)
        return self

    def predict(self, ds: Dataset):
        self._check_columns_in_ds(ds, 'feats')
        self._check_no_preds_in_ds_index(ds)
        df = ds.df
        X = df[ds.feats][self.feats]
        P = pd.DataFrame(self._core_model_predict(X), columns=self.predicts)
        new_df = pd.concat([df[ds.index], P], axis=1)
        return Dataset(df=new_df, index=ds.index, predicts=self.predicts)

    def _core_model_predict(self, X: pd.DataFrame):
        return self._core_model.predict(X)

    def init_new(self, feats=None, hyperparams=None):
        init_args = {
            arg: getattr(self, arg) 
            for arg in ['targets', 'predicts', 'core_model_class', 'name']
        }
        init_args['feats'] = self.feats if feats is None else feats
        init_args['hyperparams'] = self.hyperparams if hyperparams is None else hyperparams
        return self.__class__(**init_args)

    def _set_core_model(self, core_model_class, core_model):
        if (core_model_class is not None) and (core_model is None):
            self._core_model_class = core_model_class
            self._core_model = core_model_class(**self.hyperparams)
        elif (core_model_class is None) and (core_model is not None):
            self._core_model = copy.deepcopy(core_model)
            self._core_model_class = core_model.__class__
        else:
            raise ValueError(
                "One and only one of parameters 'core_model', 'core_model_class' should be given.")

    def _check_columns_in_ds(self, ds: Dataset, role: str):
        cols_diff = set(getattr(self, role)) - set(getattr(ds, role))
        if len(cols_diff) > 0:
            not_found_str = ", ".join([f"'{x}'" for x in cols_diff])
            raise KeyError(
                f"Model's {role} {not_found_str} are not present in the dataset's {role}")

    def _check_no_preds_in_ds_index(self, ds: Dataset):
        conflict_set = set(self.predicts) & set(ds.index)
        if len(conflict_set) > 0:
            conflict_str = ", ".join([f"'{x}'" for x in conflict_set])
            raise KeyError(
                f"Model's prediction cols {conflict_str} conflict with dataset's index")            

    @property
    def feats(self):
        return list(self._feats)

    @feats.setter
    def feats(self, val):
        self._check_list_of_str(val, 'feats')
        self._feats = list(val)

    @property
    def targets(self):
        return list(self._targets)

    @targets.setter
    def targets(self, val):
        self._check_list_of_str(val, 'targets')
        self._targets = list(val)

    @property
    def predicts(self):
        return list(self._predicts)

    @predicts.setter
    def predicts(self, val):
        self._check_list_of_str(val, 'predicts')
        self._predicts = list(val)

    def _check_list_of_str(self, x, name):
        if not isinstance(x, list):
            raise TypeError(f'Argument "{name}" must be of a type "list"')
        if not all([isinstance(s, str) for s in x]):
            raise TypeError(f'Argument "{name}" must be a list of strings')

    @property
    def hyperparams(self):
        return copy.deepcopy(self._hyperparams)

    @hyperparams.setter
    def hyperparams(self, val):
        self._hyperparams = copy.deepcopy(dict(val))

    @property
    def core_model_class(self):
        return self._core_model_class
