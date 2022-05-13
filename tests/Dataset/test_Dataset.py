import pytest
from pandas.testing import assert_frame_equal
from ...Dataset import Dataset


class TestDataset:
    
    def test_default_init(self, pandas_df):
        df = pandas_df
        ds = Dataset(df)
        assert_frame_equal(ds.df, df)
        assert(list(ds.columns) == list(ds.othercols))

    def test_init(self, pandas_df_with_col_roles):
        df, role_cols = pandas_df_with_col_roles
        ds = Dataset(df, index=role_cols['index'], features=role_cols['features'], 
            targets=role_cols['targets'], predictions=role_cols['predictions'])
        
        assert ds.columns == list(df.columns)
        
        for role in ['index', 'features', 'targets', 'predictions']:
            assert role_cols[role] == getattr(ds, role)
        
        set_othercols = set(df.columns)
        for role in ['index', 'features', 'targets', 'predictions']:
            set_othercols -= set(role_cols[role])
        assert set(ds.othercols) == set_othercols 

    @pytest.mark.parametrize('conflict_pair',
        [
            (role1, role2)
            for i, role1 in enumerate(['index', 'features', 'targets', 'predictions'])
            for j, role2 in enumerate(['index', 'features', 'targets', 'predictions'])
            if i < j
        ]
    )
    def test_xfail_roles_intersect(self, pandas_df, conflict_pair):
        df = pandas_df
        role1, role2 = conflict_pair
        with pytest.raises(ValueError):
            ds = Dataset(df, **{role1: df.columns[:2], role2: df.columns[1:]})
