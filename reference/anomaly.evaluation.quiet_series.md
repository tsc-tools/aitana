## anomaly.evaluation.quiet_series()


Generate a quiet background: a repeating template plus small noise.


Usage

``` python
anomaly.evaluation.quiet_series(
    n,
    period=144,
    noise=0.05,
    rng=None,
)
```


The template is a sine of length `period` plus a half-amplitude third harmonic, so normal behaviour recurs and every normal subsequence has close matches in the history. Similarity-based detectors (e.g. a z-normalized matrix profile) need this; on pure noise every subsequence is roughly equally far from all others.


## Parameters


`n: int`  
Number of samples.

`period: int = ``144`  
Samples per template cycle; the default is one day of 10-min samples.

`noise: float = ``0.05`  
Standard deviation of the additive Gaussian noise.

`rng: int, np.random.Generator, or None = None`  
Random seed or generator for reproducibility.


## Returns


`pd.Series`  
Synthetic series on a RangeIndex.
