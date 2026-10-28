## access.tilde.tilde_request()


Request data from the tilde API (https://tilde.geonet.org.nz/v3/api-docs/).


Usage

``` python
access.tilde.tilde_request(
    start_date,
    end_date,
    domain,
    name,
    station,
    method="-",
    sensor="-",
    aspect="-"
)
```


See the tilde discovery tool for more information: https://tilde.geonet.org.nz/ui/data-discovery/


## Parameters


`domain: str`  
The domain of the data (e.g. 'manualcollect')

`name: str`  
The name of the data (e.g. 'plume-SO2-gasflux')

`station: str`  
The station code (e.g. 'WI000')

`method: str = ``"-"`  
The method of the data (e.g. 'contouring')

`sensor: str = ``"-"`  
The sensor of the data (e.g. 'MC01')

`aspect: str = ``"-"`  
The aspect of the data (e.g. 'nil')

`start_date: date`  
The start date of the data

`end_date: date`  
The end date of the data


## Returns


`pd.DataFrame`  
A pandas dataframe with the requested data
