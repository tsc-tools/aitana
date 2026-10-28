## pipelines.scoring.make_strictly_increasing()


Make a sequence strictly increasing. Some of the ROC curves computed with


Usage

``` python
pipelines.scoring.make_strictly_increasing(sequence)
```


our own metric are not strictly increasing due to the way tps, fps, tns and fns are defined. This function makes sure that the sequence is strictly increasing so that we can caluculate the AUC value.


## Arguments:

    sequence: Sequence
        The sequence to make strictly increasing.


## Returns

    Sequence: The strictly increasing sequence.
