import apache_beam as beam
import pandas as pd
import xarray as xr

from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import (
    ConsolidateMetadata,
    Indexed,
    OpenURLWithFSSpec,
    OpenWithXarray,
    StoreToPyramid,
)

SHORT_NAME = 'GPM_3IMERGDF.07'
CONCAT_DIMS = ['time']
IDENTICAL_DIMS = ['lat', 'lon']

# 2023/07/3B-DAY.MS.MRG.3IMERG.20230731
dates = [
    d.to_pydatetime().strftime('%Y/%m/3B-DAY.MS.MRG.3IMERG.%Y%m%d')
    for d in pd.date_range('2000-06-01', '2002-06-01', freq='D')
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
        ds = ds.drop_vars('time_bnds')
        ds = ds[['precipitation']]
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._dropvarcoord)


class TransposeCoords(beam.PTransform):
    """Transform to transpose coordinates for pyramids and drop time_bnds variable"""

    @staticmethod
    def _transpose_coords(item: Indexed[xr.Dataset]) -> Indexed[xr.Dataset]:
        index, ds = item
        ds = ds.transpose('time', 'lat', 'lon', 'nv')
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._transpose_coords)



recipe = (
    beam.Create(pattern.items())
    | OpenURLWithFSSpec()
    | OpenWithXarray(file_type=pattern.file_type)
    | TransposeCoords()
    | 'Write Pyramid Levels'
    >> StoreToPyramid(
        store_name=SHORT_NAME,
        epsg_code='4326',
        rename_spatial_dims={'lon': 'longitude', 'lat': 'latitude'},
        levels=4,
        pyramid_kwargs={'extra_dim': 'nv'},
        combine_dims=pattern.combine_dim_keys,
    )
)
