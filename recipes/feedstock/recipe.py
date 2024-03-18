import io
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Tuple, List, NewType, Optional, Iterator, Union, Any, Dict

import apache_beam as beam
import pandas as pd
import xarray as xr
import fsspec
import zarr

from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.types import Index, Dimension
from pangeo_forge_recipes.writers import ZarrWriterMixin
from pangeo_forge_recipes.transforms import (
    DetermineSchema,
    IndexItems,
    Indexed,
    OpenURLWithFSSpec,
    OpenWithXarray,
    StoreToZarr
)

SHORT_NAME = 'GPM_3IMERGDF.07'
CONCAT_DIMS = ['time']
IDENTICAL_DIMS = ['lat', 'lon']

# 2023/07/3B-DAY.MS.MRG.3IMERG.20230731
dates = [
    d.to_pydatetime().strftime('%Y/%m/3B-DAY.MS.MRG.3IMERG.%Y%m%d')
    for d in pd.date_range('2000-06-01', '2001-06-01', freq='D')
]


def make_filename(time):
    base_url = f's3://gesdisc-cumulus-prod-protected/GPM_L3/{SHORT_NAME}/'
    return f'{base_url}{time}-S000000-E235959.V07B.nc4'


concat_dim = ConcatDim('time', dates, nitems_per_file=1)
pattern = FilePattern(make_filename, concat_dim)


recipe = (
    beam.Create(pattern.items())
    | FSXRFactory()
    |
)
