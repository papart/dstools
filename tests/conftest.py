import pytest
import numpy as np
import pandas as pd

@pytest.fixture()
def pandas_df():
    N = 1000
    n_feats = 5
    n_index = 2
    n_targets = 2
    n_predicts = 2
    n_othercols = 3

    np.random.seed(42)
    return pd.concat([
        pd.DataFrame(np.arange(n_index * N).reshape(N, -1), columns=[f'idx{i}' for i in range(n_index)]),
        pd.DataFrame(np.random.randn(N, n_feats), columns=[f'feat{i}' for i in range(n_feats)]),
        pd.DataFrame(np.random.randn(N, n_targets), columns=[f'target{i}' for i in range(n_targets)]),
        pd.DataFrame(np.random.randn(N, n_predicts), columns=[f'predict{i}' for i in range(n_predicts)]),
        pd.DataFrame(np.random.randn(N, n_othercols), columns=[f'other{i}' for i in range(n_othercols)]),
    ], axis=1)

@pytest.fixture()
def pandas_df_with_col_roles(pandas_df):
    df = pandas_df
    dict_role_prefix = {'index': 'idx', 'feats': 'feat', 'targets': 'target', 'predicts': 'predict'}
    col_roles = {
        role: [col for col in df.columns if col.startswith(prefix)]
        for role, prefix in dict_role_prefix.items()
    }
    return (df, col_roles)
