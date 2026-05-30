import pandas as pd

from drnb.io import read_feather, read_parquet, write_feather, write_parquet


def test_pyarrow_backed_dataframe_roundtrips(tmp_path) -> None:
    df = pd.DataFrame({"label": ["a", "b"], "value": [1.5, 2.5]})

    write_parquet(df, "roundtrip", suffix="", drnb_home=tmp_path, sub_dir="data")
    write_feather(df, "roundtrip", suffix="", drnb_home=tmp_path, sub_dir="data")

    pd.testing.assert_frame_equal(
        read_parquet("roundtrip", suffix="", drnb_home=tmp_path, sub_dir="data"), df
    )
    pd.testing.assert_frame_equal(
        read_feather("roundtrip", suffix="", drnb_home=tmp_path, sub_dir="data"), df
    )
