import pytest
from ...Dataset import Dataset

@pytest.fixture()
def dataset(pandas_df_with_col_roles):
    df, role_cols = pandas_df_with_col_roles
    ds = Dataset(df, **{role: cols for role, cols in role_cols.items()})
    return ds
