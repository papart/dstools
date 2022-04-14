import pytest
from pandas.testing import assert_frame_equal
from ...Operation import OperationInterface
from .fixtures import *


class DummyOperation(OperationInterface):

    def apply(self, ds: Dataset) -> Dataset:
        return ds

    def fit(self, ds: Dataset):
        pass


class TestOperationInterface:

    def test_xfail_create(self):
        with pytest.raises(TypeError):
            op = OperationInterface()

    @pytest.mark.parametrize('role_changed', ['index', 'feats', 'targets', 'predicts'])
    def test_recreate_dataset_change_col_roles(self, dataset, role_changed):
        ds = dataset
        op = DummyOperation()
        ds_new = op._recreate_dataset(ds, **{role_changed: ['other0']})
        for role in ['index', 'feats', 'targets', 'predicts']:
            if role != role_changed:
                assert getattr(ds, role) == getattr(ds_new, role)
        assert getattr(ds_new, role_changed) == ['other0']
        assert_frame_equal(ds_new.df, ds.df)

    def test_recreate_dataset_change_df(self, dataset):
        ds = dataset
        op = DummyOperation()
        # New dataset: drop othercols
        new_cols = [col for col in ds.columns if col not in ds.othercols]
        ds_new = op._recreate_dataset(ds, df=ds.df[new_cols])

        assert ds_new.columns == new_cols
        for role in ['index', 'feats', 'targets', 'predicts']:
            assert getattr(ds, role) == getattr(ds_new, role)
        assert ds_new.othercols == []
        assert_frame_equal(ds_new.df, ds.df[new_cols])
