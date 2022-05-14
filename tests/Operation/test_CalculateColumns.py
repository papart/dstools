import numpy as np
import pytest
from ...Operation import CalculateColumns
from ..Dataset.fixtures_Dataset import *


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
    @pytest.mark.parametrize('role', [None, 'index', 'features', 'targets', 'predictions'])
    def test_new_col(self, dataset, expr, role):
        ds = dataset
        op = CalculateColumns({'new_col': expr}, roles=role)
        ds_new = op.apply(ds)
        if callable(expr):
            expr = expr(ds.df)
        assert (ds_new.df['new_col'] == expr).all()
        if role is None:
            assert 'new_col' in ds_new.othercols
        else:
            assert 'new_col' in getattr(ds_new, role)


    @pytest.mark.parametrize('column', ['feat0'])
    @pytest.mark.parametrize('expr',
        [
            1,
            'QWER',
            lambda x: x['feat0']**2,
            lambda x: x['feat0'] + x['target0']
        ]
    )
    @pytest.mark.parametrize('role', [None, 'index', 'features', 'targets', 'predictions'])
    def test_overwrite_col(self, dataset, expr, role, column):
        ds = dataset
        op = CalculateColumns({column: expr}, roles=role)
        old_role = None
        for r in ['index', 'features', 'targets', 'predictions']:
            if column in getattr(ds, r):
                old_role = r

        ds_new = op.apply(ds)
        if callable(expr):
            expr = expr(ds.df)
        assert (ds_new.df[column] == expr).all()
        if role is None:
            assert column in getattr(ds_new, old_role)
        else:
            assert column in getattr(ds_new, role)
