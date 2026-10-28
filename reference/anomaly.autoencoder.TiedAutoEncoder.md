## anomaly.autoencoder.TiedAutoEncoder


Applies an tied-weight autoencoder to the incoming data.


Usage

``` python
anomaly.autoencoder.TiedAutoEncoder(
    in_features,
    h_features,
    activation=F.relu,
)
```


From https://gist.github.com/northanapon/375e17fb395391c144deff20914e51df


## Parameters


`in_features: int`  
size of each input sample.

`h_features: List[int]`  
a list of size of each layer for the encoder and the reverse for the decoder.

`activation: function = F.relu`  
an activation function to apply for each layer (the default is torch.tanh).


## Methods

| Name | Description |
|----|----|
| [__init__()](#__init__) | Create an autoencoder. |
| [decode()](#decode) | Return decoded data. |
| [encode()](#encode) | Return encoded data. |
| [forward()](#forward) | Return result of encoding and decoding. |
| [reset_parameters()](#reset_parameters) | Reset linear module parameters (from nn.Linear). |

------------------------------------------------------------------------


#### \_\_init\_\_()


Create an autoencoder.


Usage

``` python
__init__(in_features, h_features, activation=F.relu)
```


------------------------------------------------------------------------


#### decode()


Return decoded data.


Usage

``` python
decode(o)
```


------------------------------------------------------------------------


#### encode()


Return encoded data.


Usage

``` python
encode(x)
```


------------------------------------------------------------------------


#### forward()


Return result of encoding and decoding.


Usage

``` python
forward(x)
```


------------------------------------------------------------------------


#### reset_parameters()


Reset linear module parameters (from nn.Linear).


Usage

``` python
reset_parameters()
```
