import os

import numpy as np
import pytest

from aitana.anomaly.anomaly_base import MinMaxScale, min_max_scale_netcdf

# setup_dataset scaled with MinMaxScale: one row per time step, columns rsam
# and the three ssam frequencies; gaps are filled with the median first
SCALED = np.array(
    [
        [0.0, 0.0, 0.0, 1 / 2.0],
        [1 / 3.0, 1 / 2.0, 1 / 3.0, 0.0],
        [1 / 3.0, 1 / 2.0, 1 / 3.0, 1 / 2.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)


def test_min_max_scaling(setup_dataset):
    sg = setup_dataset
    for feature in ["rsam"]:
        fin = sg.feature_path(feature)
        fout = os.path.join(sg.path, f"{feature}_scaling_params.nc")
        scaling_params = min_max_scale_netcdf(fin, feature, fout)
        assert scaling_params["min"] == 4.0
        assert scaling_params["max"] == 7.0
        assert scaling_params["median"] == 5.0

    for feature in ["ssam"]:
        fin = sg.feature_path(feature)
        fout = os.path.join(sg.path, f"{feature}_scaling_params.nc")
        scaling_params = min_max_scale_netcdf(fin, feature, fout)
        assert scaling_params["min"] == [1.0, 4.0, 9.0]
        assert scaling_params["max"] == [3.0, 7.0, 11.0]
        assert scaling_params["median"] == [2.0, 5.0, 10.0]


def test_min_max_scale(setup_dataset):
    sg = setup_dataset
    block = np.column_stack([sg("rsam").values, sg("ssam").values.T])
    scaled = MinMaxScale(sg, ["rsam", "ssam"])(block.astype(np.float32))
    np.testing.assert_allclose(scaled, SCALED, rtol=1e-6)


def test_dataset_with_scaling(setup_dataset):
    pytest.importorskip("torch")
    from aitana.storage.storage import AitanaDataset

    sg = setup_dataset
    features = ["rsam", "ssam"]
    ds = AitanaDataset(sg, features, transform=MinMaxScale(sg, features))
    assert ds.n_features == 4
    assert len(ds) == 4
    samples = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    np.testing.assert_allclose(samples, SCALED, rtol=1e-6)
