import apache_beam as beam
import pandas as pd
import os
from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import (
    OpenURLWithFSSpec,
    OpenWithXarray,
    StoreToZarr,
)

dates = [
    d.to_pydatetime().strftime('_%Y_%m_3b-day.ms.mrg.3imerg.%Y%m%d')
    for d in pd.date_range('2000-06-01', '2005-06-01', freq='D')
]

def make_filename(time):
    # example:
    # s3://pangeo-forge-uah-test-output/inputs/s3_gesdisc-cumulus-prod-protected_gpm_l3_gpm_3imergdf.07_2001_01_3b-day.ms.mrg.3imerg.20010125-s000000-e235959.v07b.nc4
    base_url = "s3://pangeo-forge-uah-test-output/inputs/s3_gesdisc-cumulus-prod-protected_gpm_l3_gpm_3imergdf.07"
    return f'{base_url}{time}-s000000-e235959.v07b.nc4'

concat_dim = ConcatDim('time', dates, nitems_per_file=1)
pattern = FilePattern(make_filename, concat_dim)

fsspec_kwargs = {
    "key": os.environ.get("AWS_ACCESS_KEY_ID"),
    "secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "anon": True
}

recipe = (
    beam.Create(pattern.items())
    | OpenURLWithFSSpec(open_kwargs=fsspec_kwargs)
    | OpenWithXarray()
    | StoreToZarr(
        store_name="test.zarr",
        combine_dims=pattern.combine_dim_keys,
    )
)
