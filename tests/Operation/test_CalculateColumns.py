import numpy as np
from ...Operation import CalculateColumns
from .fixtures import *


class TestCalculateColumns:


    @pytest.mark.parametrize('expr',
        [
            1,
            'QWER',
            # np.nan,
            lambda x: x['feat0']**2,
            lambda x: x['feat0'] + x['target0']
        ]
    )
    @pytest.mark.parametrize('role', [None, 'index', 'feats', 'targets', 'predicts'])
    def test_new_col(self, dataset, expr, role):
        ds = dataset
        op = CalculateColumns({'new_col': expr}, roles=role)
        ds_new = op.apply(ds)
        if callable(expr):
            expr = expr(ds.df)
        assert (ds_new.df['new_col'] == expr).all()
