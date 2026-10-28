## pipelines.cli.run()


Run the workflow registered for a volcano.


Usage

``` python
pipelines.cli.run(
    ctx,
    volcano,
    outdir,
    cores=1,
    cron=None,
    workflow=None,
)
```


With several workflows registered for the volcano, choose one with `--workflow NAME` (see `list`); each needs its own output directory. Extra arguments are passed to Snakemake, e.g. `--config end=2022-03-20T00:00`. With `--cron`, keep running and repeat on a crontab schedule in UTC, e.g. `--cron "*/10 * * * *"`; a run that is due while the previous one is still going is skipped.
