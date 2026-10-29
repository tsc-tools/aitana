import json
import os

import numpy as np
import xarray as xr


def min_max_scale_netcdf(
    input_file, variable_name, scaling_params_file, chunk_size=144
):
    if os.path.isfile(scaling_params_file):
        with open(scaling_params_file, "r") as f:
            scaling_params = json.load(f)
        return scaling_params

    with xr.open_dataset(input_file, group="original") as ds:
        data = ds[variable_name]
        if len(data.sizes) > 1:
            min_value = list(data.min(axis=1).values)
            max_value = list(data.max(axis=1).values)
            median_value = list(data.median(axis=1).values)
        else:
            min_value = float(data.min())
            max_value = float(data.max())
            median_value = float(data.median())
    scaling_params = {"min": min_value, "max": max_value, "median": median_value}
    with open(scaling_params_file, "w") as f:
        json.dump(scaling_params, f)

    return scaling_params


class MinMaxScale:
    """
    Fill gaps with the median and scale each feature column to [0, 1].

    A ``transform`` for ``aitana.storage.storage.AitanaDataset``. Minimum,
    maximum and median of each feature come from its whole archive, computed
    once with `min_max_scale_netcdf` and cached as
    ``scaling_params_<feature>.json`` in the store.

    Parameters
    ----------
    store : aitana.storage.Path
        Store holding the features, in NetCDF format.
    features : list of str
        Feature names, in the order given to the dataset.
    """

    def __init__(self, store, features):
        params = [
            min_max_scale_netcdf(
                store.feature_path(feature),
                feature,
                os.path.join(store.path, f"scaling_params_{feature}.json"),
            )
            for feature in features
        ]

        def columns(key):
            # One value per column; 2D features have one per row of their archive
            return np.concatenate([np.atleast_1d(p[key]) for p in params]).astype(
                np.float32
            )

        self.min = columns("min")
        self.max = columns("max")
        self.median = columns("median")

    def __call__(self, block):
        """
        Scale a block of time steps.

        Parameters
        ----------
        block : np.ndarray
            Array of shape ``[n_timesteps, n_features]``, with columns in the
            dataset's order.

        Returns
        -------
        np.ndarray
            The scaled block; NaN are replaced by the median before scaling.
        """
        block = np.where(np.isnan(block), self.median, block)
        return (block - self.min) / (self.max - self.min)


class AnomalyDetectionBaseClass(object):
    def __init__(self, features, stacks=None, transforms=None):
        self.features = features
        self.stacks = stacks if stacks is not None else dict()
        self.transforms = transforms if transforms is not None else dict()

    def fit(self, starttime, endtime):
        """
        Train the model.
        """
        msg = "'training' is not implemented yet"
        raise NotImplementedError(msg)

    def infer(self, starttime, endtime):
        """
        Use the model to infer.
        """
        msg = "'inference' is not implemented yet"
        raise NotImplementedError(msg)

    def write_model_parameters(self):
        """
        Write principal components or
        model weights to file
        """
        msg = "'write_model_parameters' is not implemented yet"
        raise NotImplementedError(msg)

    def read_model_parameters(self):
        """
        Read principal components or
        model weights from file
        """
        msg = "'read_model_parameters' is not implemented yet"
        raise NotImplementedError(msg)

    def write_hyperparameters(self):
        """
        Store things like random seed,
        learning rate, number of principal
        components etc.
        """
        msg = "'write_hyperparameters' is not implemented yet"
        raise NotImplementedError(msg)

    def compute_anomaly_index(self):
        """
        A scalar value between 0 and 1 indicating
        the degree of anomaly (1=most anomaluous).
        """
        msg = "'compute_anomaly_index' is not implemented yet"
        raise NotImplementedError(msg)

    @staticmethod
    def normalize_score(raw_scores, reference_scores):
        """
        Percentile rank (ECDF) of raw_scores against a reference distribution.

        Parameters
        ----------
        raw_scores : array-like
            Scores to normalize (e.g. from the evaluation period).
        reference_scores : array-like
            Reference distribution from a quiet period.

        Returns
        -------
        np.ndarray
            Values in [0, 1]; 1.0 means the score exceeds all reference scores.
        """
        raw = np.asarray(raw_scores, dtype=float)
        ref = np.sort(np.asarray(reference_scores, dtype=float))
        return np.searchsorted(ref, raw, side="right") / len(ref)
