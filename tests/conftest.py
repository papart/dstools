import pytest
import numpy as np
import pandas as pd

@pytest.fixture()
def pandas_df():
    N = 1000
    n_features = 5
    n_index = 2
    n_targets = 2
    n_predictions = 2
    n_othercols = 3

    np.random.seed(42)
    return pd.concat([
        pd.DataFrame(np.arange(n_index * N).reshape(N, -1), columns=[f'idx{i}' for i in range(n_index)]),
        pd.DataFrame(np.random.randn(N, n_features), columns=[f'feat{i}' for i in range(n_features)]),
        pd.DataFrame(np.random.randn(N, n_targets), columns=[f'target{i}' for i in range(n_targets)]),
        pd.DataFrame(np.random.randn(N, n_predictions), columns=[f'predict{i}' for i in range(n_predictions)]),
        pd.DataFrame(np.random.randn(N, n_othercols), columns=[f'other{i}' for i in range(n_othercols)]),
    ], axis=1)

@pytest.fixture()
def pandas_df_with_col_roles(pandas_df):
    df = pandas_df
    dict_role_prefix = {'index': 'idx', 'features': 'feat', 'targets': 'target', 'predictions': 'predict'}
    col_roles = {
        role: [col for col in df.columns if col.startswith(prefix)]
        for role, prefix in dict_role_prefix.items()
    }
    return (df, col_roles)
