## storage.grafana_annotations.sync_labels()


Bring the label annotations in Grafana and the stored labels in step.


Usage

``` python
storage.grafana_annotations.sync_labels(
    baseurl,
    stored,
)
```


The result is the union of both. A label in both takes its text, tags and times from Grafana; labels drawn in Grafana, on any dashboard, are added; stored labels missing from Grafana are posted as organization annotations, so nothing is lost when an annotation is deleted or Grafana is redeployed. Annotations without an "Id: " line get a new id appended to their text.


## Parameters


`baseurl: str`  
The base URL of the Grafana API, e.g. "http://:". Authenticate with the environment variables GRAFANA_USER and GRAFANA_PASSWORD, or with "http://:@:".

`stored: list of dict`  
Labels from the previous sync, as returned by this function.


## Returns


`labels: list of dict`  
All labels, ordered by time. Each has 'time' and 'timeEnd' (ms since epoch; 'timeEnd' is None for a point), 'title' (first line of the text), 'description' (the text without its "Id:" line), 'tags', 'id' (the stable label id), and where they were made: 'user' (login), 'dashboardUID' and 'panelId' (None for an annotation made through the API).

`report: dict`  
Number of labels 'restored' to Grafana and 'added' from Grafana, and 'untagged': the labels without a `volcano:` tag, which no volcano dashboard shows.
