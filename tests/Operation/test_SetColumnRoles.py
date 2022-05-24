import numpy as np
import pytest
from ...Operation import SetColumnRoles
from ..Dataset.fixtures_Dataset import *

class TestCalculateColumns:

    @pytest.mark.parametrize('new_roles',
        [
            {'feat0': 'othercols', 'predict0': 'othercols'},
            {'other0': 'features', 'other1': 'targets'},
        ]
    )
    def test_rules_constant(self, dataset, new_roles):
        ds = dataset
        op = SetColumnRoles(new_roles)
        ds_new = op.apply(ds)
        # assert that new_roles have been assigned
        for col, role in new_roles.items():
            assert col in getattr(ds_new, role)
        # assert that old roles haven't changed
        for old_role, cols in ds.get_col_roles.items():
            for col in cols:
                if col not in new_roles.keys():
                    assert col in getattr(ds_new, old_role)

    @pytest.mark.parametrize('new_roles_lambda',
        [
            lambda x: {col: 'features' for col in x.othercols},
            lambda x: {col: 'othercols' for col in x.columns if x.endswith('0')},
        ]
    )
    def test_rules_lambda(self, dataset, new_roles_lambda):
        ds = dataset
        op = SetColumnRoles(new_roles_lambda)
        ds_new = op.apply(ds)
        new_roles = new_roles_lambda(ds)
        # assert that new_roles have been assigned
        for col, role in new_roles.items():
            assert col in getattr(ds_new, role)
        # assert that old roles haven't changed
        for old_role, cols in ds.get_col_roles.items():
            for col in cols:
                if col not in new_roles.keys():
                    assert col in getattr(ds_new, old_role)

    @pytest.mark.parametrize('new_roles',
        [
            {'other0': 'targets', 'other1': 'feats'},
            lambda x: {col: 'preds' for col in x.othercols}
        ]
    )
    def test_xfail_wrong_role(self, new_roles):
        with pytest.raises(TypeError):
            op = SetColumnRoles(new_roles)

    @pytest.mark.parametrize('new_roles',
        [
            {'non_existing_column': 'targets'},
            lambda x: {'qwerty_' + col: 'index' for col in x.othercols}
        ]
    )
    def test_xfail_unknown_column(self, dataset, new_roles):
        ds = dataset
        op = SetColumnRoles(new_roles)
        with pytest.raises(ValueError):
            ds_new = op.apply(ds)
