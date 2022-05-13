from dstools.Dataset.Dataset import Dataset
from dstools.Operation.OperationInterface import OperationInterface
import pandas as pd

class CalculateColumns(OperationInterface):
    def __init__(self, expressions, roles=None, name='CalculateColumn'):
        # expressions - dict(column_name: value).
        # value is either a pd.Series, or a function: fn(ds: Dataset) -> pd.Series
        self.expressions = expressions
        self.roles = roles
        super().__init__(name=name)

    def apply(self, ds: Dataset) -> Dataset:
        df_new = ds.df # this returns a copy
        for column, expr in self.expressions.items():
            if callable(expr):
                expr = expr(ds) # calculate on the old (!) dataframe
            if not isinstance(expr, pd.Series):
                raise TypeError(
                    f"Calculated expression for column {column} has type {type(expr)}, "
                    "but must be a pandas Series")
            df_new[column] = expr
        new_col_roles = self._get_new_col_roles(ds)
        return self._recreate_dataset(ds, df=df_new, **new_col_roles)

    def fit(self, ds: Dataset):
        pass

    def _get_new_col_roles(self, ds):
        # TODO: refactoring
        col_roles = {
            role: getattr(ds, role) 
            for role in ['index', 'features', 'targets', 'predictions']
        }
        if (self.roles is None) or (isinstance(self.roles, str)):
            assigned_roles_dict = {col: self.roles for col in self.expressions.keys()}
        else:
            assigned_roles_dict = self.roles
        for col, assigned_role in assigned_roles_dict.items():
            if assigned_role is None:
                continue
            old_role = None
            for role, cols_list in col_roles.items():
                if col in cols_list:
                    old_role = role
                    break
            if old_role == assigned_role:
                continue
            if old_role is not None:
                col_roles[old_role].remove(col)
            col_roles[assigned_role].append(col)
        return col_roles

    # TODO: docstrings
    # TODO: check if roles has a valid value
    # TODO: check types in expressions
