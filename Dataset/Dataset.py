import pandas as pd

class Dataset:
    def __init__(self, 
            df, 
            index=None, feats=None, targets=None, predicts=None
        ):
        self._check_df(df)
        self._df = df.copy()
        self.set_col_roles(
            index=index, feats=feats, targets=targets, predicts=predicts)

    @property
    def df(self):
        return self._df.copy()

    def set_df(self, val):
        self._check_df(val)
        new_columns = list(val.columns)
        self._check_col_roles_consistent(columns=new_columns)
        self._df = val    

    def set_col_roles(self, **col_roles):
        for role in ['index', 'feats', 'targets', 'predicts']:
            if col_roles[role] is None:
                col_roles[role] = []
        self._check_col_roles_consistent(**col_roles)
        for role in ['index', 'feats', 'targets', 'predicts']:
            if role in col_roles:
                setattr(self, f'_{role}', col_roles[role])

    @property
    def columns(self):
        return list(self._df.columns)
    
    @property
    def index(self):
        return list(self._index)

    @property
    def feats(self):
        return list(self._feats)

    @property
    def targets(self):
        return list(self._targets)

    @property
    def predicts(self):
        return list(self._predicts)

    @property
    def othercols(self):
        othercols = [
            col for col in self._df.columns
            if not any([
                col in getattr(self, f'_{role}') 
                for role in ['index', 'feats', 'targets', 'predicts']
            ])
        ]
        return othercols

    def get_col_roles(self):
        return {
            role: getattr(self, role)
            for role in ['columns', 'index', 'feats', 'targets', 'predicts', 'othercols']
        }

    def _check_col_roles_consistent(self, **col_roles):
        main_roles = ['index', 'feats', 'targets', 'predicts']
        for role in main_roles + ['columns']:
            if role not in col_roles:
                col_roles[role] = getattr(self, role)
            col_roles[role] = set(col_roles[role])
        # Check that every column from main role is indeed in columns
        for role in main_roles:
            if not (col_roles[role] <= col_roles['columns']):
                raise ValueError(f'{role} is not a subset of columns')
        # Check that main column roles do not intersect
        for i, role1 in enumerate(main_roles):
            for role2 in main_roles[i + 1:]:
                if len(col_roles[role1] & col_roles[role2]) > 0:
                    raise ValueError(f"Column sets of '{role1}' and '{role2}' intersect")

    def _check_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be an instance of pandas.DataFrame')

    # TODO: implement subscripting: ds['column'] (optimization)
    # TODO: docstrings
    # TODO: a list of assignable roles
    # TODO: check for duplicates in index, feats, etc.
