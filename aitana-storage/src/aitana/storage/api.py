import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Annotated, List, Union, Optional
from urllib.parse import unquote

import datashader as dsh
import numpy as np
import pandas as pd
import uvicorn
from cftime import date2num, num2pydate
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from . import get_data
from .storage import Storage

logger = logging.getLogger(__name__)

InventoryReturnType = Union[list, dict]


def parse_path(path: str):
    """Split a comma-separated path string into (group, subdir_list)."""
    parts = [p.strip() for p in path.split(',') if p.strip()]
    if not parts:
        raise HTTPException(status_code=400, detail="path parameter is required and must not be empty")
    return parts[0], parts[1:]


class TonikAPI:

    def __init__(self, rootdir, backend='netcdf') -> None:
        self.rootdir = rootdir
        self.backend = backend
        self.app = FastAPI()

        # -- allow any origin to query API
        self.app.add_middleware(CORSMiddleware,
                                allow_origins=["*"])

        self.app.get("/", response_class=HTMLResponse)(self.root)
        self.app.get("/feature")(self.feature)
        self.app.get("/inventory")(self.inventory)

    def root(self):
        with open(get_data("package_data/index.html"), "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)

    def preprocess_datetime(self, dt):
        """
        Convert datetime string to datetime object.
        """
        # remove timezone info
        dt = dt.split('+')[0]
        # remove 'Z' at the end
        dt = dt.replace('Z', '')
        # convert html encoded characters
        dt = unquote(dt)
        dt = datetime.fromisoformat(dt)
        dt = dt.replace(tzinfo=None)
        return dt

    async def feature(self,
                      path: str,
                      name: str,
                      starttime: Union[str, None],
                      endtime: Union[str, None],
                      resolution: str = 'full',
                      verticalres: int = 10,
                      log: bool = False,
                      normalise: bool = False):
        group, subdir = parse_path(path)
        _st = self.preprocess_datetime(starttime)
        _et = self.preprocess_datetime(endtime)
        try:
            g = Storage(group, rootdir=self.rootdir,
                        starttime=_st, endtime=_et,
                        create=False, backend=self.backend)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Group {group} not found.")
        c = g
        if subdir:
            c = g.get_substore(*subdir)
        try:
            feat = c(name)
        except ValueError as e:
            msg = f"Feature {name} not found in directory {c.path}:"
            msg += f"{e}"
            raise HTTPException(status_code=404, detail=msg)
        if len(feat.shape) > 1:
            # assume first dimension is frequency
            nfreqs = feat.shape[0]
            dates = feat.coords[feat.dims[1]].values
            if resolution != 'full':
                freq, dates, spec = self.aggregate_feature(
                    resolution, verticalres, feat, nfreqs, dates)
            else:
                spec = feat.values
                freq = feat.coords[feat.dims[0]].values
            vals = spec.ravel(order='C')
            if log and feat.name != 'sonogram':
                vals = 10*np.log10(vals)
            if normalise:
                vals = (vals - np.nanmin(vals)) / \
                    (np.nanmax(vals) - np.nanmin(vals))
            freqs = freq.repeat(dates.size)
            dates = np.tile(dates, freq.size)
            df = pd.DataFrame(
                {'dates': dates, 'freqs': freqs, 'feature': vals})
            df['dates'] = pd.to_datetime(df.dates.values).strftime(
                '%Y-%m-%dT%H:%M:%SZ')
            output = df.to_csv(index=False,
                               columns=['dates', 'freqs', 'feature'])
        else:
            # NOTE: `log` and `normalise` are currently ignored for 1D
            # features; they are only applied in the 2D branch above.
            df = pd.DataFrame(data=feat.to_pandas(), columns=[feat.name])
            df['dates'] = df.index
            if resolution != 'full':
                try:
                    current_resolution = pd.Timedelta(
                        df['dates'].diff().mean())
                    if current_resolution < pd.Timedelta(resolution):
                        df = df.resample(pd.Timedelta(resolution)).median()
                except ValueError:
                    logger.warning(
                        f"Cannot resample {feat.name} to {resolution}: e")
            df.rename(columns={feat.name: 'feature'}, inplace=True)
            df['dates'] = df['dates'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            output = df.to_csv(index=False, columns=['dates', 'feature'])
        return StreamingResponse(iter([output]),
                                 media_type='text/csv',
                                 headers={"Content-Disposition":
                                          "attachment;filename=<tonik_feature>.csv",
                                          'Content-Length': str(len(output))})

    def aggregate_feature(self, resolution, verticalres, feat, nfreqs, dates):
        resolution = np.timedelta64(
            pd.Timedelta(resolution), 'ms').astype(float)
        ndays = np.timedelta64(dates[-1] - dates[0], 'ms').astype(float)
        canvas_x = int(ndays/resolution)
        canvas_y = min(nfreqs, verticalres)
        dates = date2num(dates.astype('datetime64[us]').astype(datetime),
                         units='hours since 1970-01-01 00:00:00.0',
                         calendar='gregorian')
        feat = feat.assign_coords({'datetime': dates})
        cvs = dsh.Canvas(plot_width=canvas_x,
                         plot_height=canvas_y)
        agg = cvs.raster(source=feat)
        freq_dim = feat.dims[0]
        freq, d, spec = agg.coords[freq_dim].values, agg.coords['datetime'].values, agg.data
        dates = num2pydate(
            d, units='hours since 1970-01-01 00:00:00.0', calendar='gregorian')
        return freq, dates, spec

    async def inventory(self, request: Request, path: Optional[str] = None, tree: bool = True) -> InventoryReturnType:
        if path is None:
            return sorted(d for d in os.listdir(self.rootdir)
                         if os.path.isdir(os.path.join(self.rootdir, d)))
        group, subdir = parse_path(path)
        try:
            sg = Storage(group, rootdir=self.rootdir,
                         create=False, backend=self.backend)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Group {group} not found.")
        try:
            c = sg.get_substore(*subdir) if subdir else sg
        except TypeError:
            c = sg
        except FileNotFoundError:
            msg = "Directory {} not found.".format(
                '/'.join([sg.path] + list(subdir)))
            raise HTTPException(status_code=404, detail=msg)
        base_url = str(request.base_url)
        if tree and not subdir:
            return sg.to_dict_with_info(base_url)
        else:
            dir_contents = os.listdir(c.path)
            result = []
            for fn in sorted(dir_contents):
                full_path = os.path.join(c.path, fn)
                if fn.endswith('.nc') or (fn.endswith('.zarr') and os.path.isdir(full_path)):
                    feature_name = fn.replace('.nc', '').replace('.zarr', '')
                    info = c.get_feature_info(feature_name)
                    rel = os.path.relpath(c.path, os.path.join(self.rootdir, group))
                    subdirs_list = rel.split(os.sep) if rel != '.' else []
                    path_str = ','.join([group] + subdirs_list)
                    info['url'] = None if info['recordCount'] == 0 else (
                        f"{base_url}feature?path={path_str}&name={feature_name}"
                        f"&starttime={info['earliestRecord']}&endtime={info['latestRecord']}"
                    )
                    result.append(info)
                else:
                    result.append(fn)
            return result


def main(argv=None):
    parser = ArgumentParser()
    parser.add_argument("--rootdir", default='/tmp')
    parser.add_argument("--backend", default='netcdf')
    parser.add_argument("-p", "--port", default=8003, type=int)
    parser.add_argument("--host", default='0.0.0.0')
    args = parser.parse_args(argv)
    ta = TonikAPI(args.rootdir, backend=args.backend)
    uvicorn.run(ta.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
