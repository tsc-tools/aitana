## anomaly.anomaly_base.AnomalyDetectionBaseClass


Usage

``` python
anomaly.anomaly_base.AnomalyDetectionBaseClass(
    features, stacks=None, transforms=None
)
```


## Methods

| Name | Description |
|----|----|
| [compute_anomaly_index()](#compute_anomaly_index) | A scalar value between 0 and 1 indicating |
| [fit()](#fit) | Train the model. |
| [infer()](#infer) | Use the model to infer. |
| [normalize_score()](#normalize_score) | Percentile rank (ECDF) of raw_scores against a reference distribution. |
| [read_model_parameters()](#read_model_parameters) | Read principal components or |
| [write_hyperparameters()](#write_hyperparameters) | Store things like random seed, |
| [write_model_parameters()](#write_model_parameters) | Write principal components or |

------------------------------------------------------------------------


#### compute_anomaly_index()


A scalar value between 0 and 1 indicating


Usage

``` python
compute_anomaly_index()
```


the degree of anomaly (1=most anomaluous).


------------------------------------------------------------------------


#### fit()


Train the model.


Usage

``` python
fit(starttime, endtime)
```


------------------------------------------------------------------------


#### infer()


Use the model to infer.


Usage

``` python
infer(starttime, endtime)
```


------------------------------------------------------------------------


#### normalize_score()


Percentile rank (ECDF) of raw_scores against a reference distribution.


Usage

``` python
normalize_score(raw_scores, reference_scores)
```


##### Parameters


`raw_scores: array - like`  
Scores to normalize (e.g. from the evaluation period).

`reference_scores: array - like`  
Reference distribution from a quiet period.


##### Returns


`np.ndarray`  
Values in \[0, 1\]; 1.0 means the score exceeds all reference scores.


------------------------------------------------------------------------


#### read_model_parameters()


Read principal components or


Usage

``` python
read_model_parameters()
```


model weights from file


------------------------------------------------------------------------


#### write_hyperparameters()


Store things like random seed,


Usage

``` python
write_hyperparameters()
```


learning rate, number of principal components etc.


------------------------------------------------------------------------


#### write_model_parameters()


Write principal components or


Usage

``` python
write_model_parameters()
```


model weights to file
