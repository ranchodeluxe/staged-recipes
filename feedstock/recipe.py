import apache_beam as beam
import pandas as pd
import xarray as xr
import fsspec
import s3fs
import boto3

from beam_pyspark_runner.pyspark_runner import PySparkRunner
from pangeo_forge_recipes.storage import FSSpecTarget
from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import (
    Indexed,
    OpenURLWithFSSpec,
    OpenWithXarray,
    StoreToZarr,
)

import logging
logging.getLogger().setLevel(logging.DEBUG)

SHORT_NAME = 'GPM_3IMERGDF.07'
CONCAT_DIMS = ['time']
IDENTICAL_DIMS = ['lat', 'lon']

# 2023/07/3B-DAY.MS.MRG.3IMERG.20230731
dates = [
    d.to_pydatetime().strftime('%Y/%m/3B-DAY.MS.MRG.3IMERG.%Y%m%d')
    for d in pd.date_range('2000-06-01', '2005-06-01', freq='D')
]


def make_filename(time):
    base_url = f's3://gesdisc-cumulus-prod-protected/GPM_L3/{SHORT_NAME}/'
    return f'{base_url}{time}-S000000-E235959.V07B.nc4'


concat_dim = ConcatDim('time', dates, nitems_per_file=1)
pattern = FilePattern(make_filename, concat_dim)


class DropVarCoord(beam.PTransform):
    """Drops non-viz variables & time_bnds."""

    @staticmethod
    def _dropvarcoord(item: Indexed[xr.Dataset]) -> Indexed[xr.Dataset]:
        index, ds = item
        # Removing time_bnds since it doesn't have spatial dims
        ds = ds.drop_vars('time_bnds')  # b/c it points to nv dimension
        ds = ds[['precipitation']]
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._dropvarcoord)


class TransposeCoords(beam.PTransform):
    """Transform to transpose coordinates for pyramids and drop time_bnds variable"""

    @staticmethod
    def _transpose_coords(item: Indexed[xr.Dataset]) -> Indexed[xr.Dataset]:
        index, ds = item
        ds = ds.transpose('time', 'lat', 'lon')
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._transpose_coords)


def print_and_return(x):
    print(x)
    return x

# NOTE: source uses the EMR serverless execution role (veda-data-reader-dev)
source_fsspec_kwargs = {
  'anon': False,
  'client_kwargs': {'region_name': 'us-west-2'},
}

# NOTE: target uses the EMR serverless execution role (veda-data-reader-dev)
target_fsspec_kwargs = {
	"anon": False,
	"client_kwargs": {"region_name": "us-west-2"}
}
fs_target = s3fs.S3FileSystem(**target_fsspec_kwargs)
#fs_target = fsspec.implementations.local.LocalFileSystem()
target_root = FSSpecTarget(fs_target, 's3://veda-pforge-emr-outputs-v3')
#target_root = FSSpecTarget(fs_target, '/home/jovyan/outputs/')


with beam.Pipeline(runner=PySparkRunner()) as p:
    (p | beam.Create(pattern.items())
	| OpenURLWithFSSpec(open_kwargs=source_fsspec_kwargs)
	| OpenWithXarray(file_type=pattern.file_type)
	| StoreToZarr(
        target_root=target_root,
		store_name="gpm5yr.zarr",
		combine_dims=pattern.combine_dim_keys,
	))

