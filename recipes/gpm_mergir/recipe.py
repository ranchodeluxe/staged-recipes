import base64
import json
import zarr
import os
from dataclasses import dataclass, field
from typing import Dict, Union

import apache_beam as beam
import requests
from cmr import GranuleQuery
from kerchunk.combine import MultiZarrToZarr
from xarray import Dataset

from pangeo_forge_recipes.patterns import pattern_from_file_sequence
from pangeo_forge_recipes.storage import FSSpecTarget
from pangeo_forge_recipes.transforms import (
    CombineReferences,
    OpenWithKerchunk,
    RequiredAtRuntimeDefault,
    WriteCombinedReference,
)
from pangeo_forge_recipes.writers import ZarrWriterMixin

HTTP_REL = 'http://esipfed.org/ns/fedsearch/1.1/data#'
S3_REL = 'http://esipfed.org/ns/fedsearch/1.1/s3#'

# This recipe requires the following environment variables from Earthdata
ED_TOKEN = os.environ['EARTHDATA_TOKEN']
ED_USERNAME = os.environ['EARTHDATA_USERNAME']
ED_PASSWORD = os.environ['EARTHDATA_PASSWORD']

CREDENTIALS_API = 'https://data.gesdisc.earthdata.nasa.gov/s3credentials'
SHORT_NAME = 'GPM_MERGIR'
CONCAT_DIM = 'time'
IDENTICAL_DIMS = ['lat', 'lon']

# use HTTP_REL if S3 access is not possible. S3_REL is faster.
selected_rel = S3_REL


def earthdata_auth(username: str, password: str):
    login_resp = requests.get(CREDENTIALS_API, allow_redirects=False)
    login_resp.raise_for_status()

    encoded_auth = base64.b64encode(f'{username}:{password}'.encode('ascii'))
    auth_redirect = requests.post(
        login_resp.headers['location'],
        data={'credentials': encoded_auth},
        headers={'Origin': CREDENTIALS_API},
        allow_redirects=False,
    )
    auth_redirect.raise_for_status()

    final = requests.get(auth_redirect.headers['location'], allow_redirects=False)

    results = requests.get(CREDENTIALS_API, cookies={'accessToken': final.cookies['accessToken']})
    results.raise_for_status()

    creds = json.loads(results.content)
    print(creds)
    return {
        'aws_access_key_id': creds['accessKeyId'],
        'aws_secret_access_key': creds['secretAccessKey'],
        'aws_session_token': creds['sessionToken'],
    }


def filter_data_links(links, rel):
    return filter(
        lambda link: link['rel'] == rel
        and (link['href'].endswith('.nc') or link['href'].endswith('.nc4')),
        links,
    )


def gen_data_links(rel):
    granules = GranuleQuery().short_name(SHORT_NAME).downloadable(True).get_all()
    count = 0
    for granule in granules:
        s3_links = filter_data_links(granule['links'], rel)
        first = next(s3_links, None)
        # throw if CMR does not have exactly one S3 link for an item
        if not first or next(s3_links, None) is not None:
            raise ValueError(f"Expected 1 link of type {rel} on {granule['title']}")
        print(first)
        yield first['href']
        count += 1
        if count >= 5000:
            return


@dataclass
class ConsolidateMetadata(beam.PTransform):
    """Consolidate metadata into a single .zmetadata file.

    See zarr.consolidate_metadata() for details.
    """

    storage_options: Dict = field(default_factory=dict)

    @staticmethod
    def _consolidate(mzz: MultiZarrToZarr, storage_options: Dict) -> Dict:
        import fsspec
        import zarr
        from kerchunk.utils import consolidate

        out = mzz.translate()
        fs = fsspec.filesystem(
            'reference',
            fo=out,
            remote_options=storage_options,
        )
        mapper = fs.get_mapper()
        metadata_key = '.zmetadata'
        zarr.consolidate_metadata(mapper, metadata_key=metadata_key)
        double_consolidated = consolidate(dict([(metadata_key, mapper[metadata_key])]))
        out['refs'] = out['refs'] | double_consolidated['refs']
        return out

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._consolidate, storage_options=self.storage_options)


@dataclass
class ValidateDatasetDimensions(beam.PTransform):
    """Open the reference.json in xarray and validate dimensions."""

    expected_dims: Dict = field(default_factory=dict)

    @staticmethod
    def _validate(ds: Dataset, expected_dims: Dict) -> None:
        if set(ds.dims) != expected_dims.keys():
            raise ValueError(f'Expected dimensions {expected_dims.keys()}, got {ds.dims}')
        for dim, bounds in expected_dims.items():
            if bounds is None:
                continue
            lo, hi = bounds
            actual_lo, actual_hi = round(ds[dim].data.min()), round(ds[dim].data.max())
            if actual_lo != lo or actual_hi != hi:
                raise ValueError(f'Expected {dim} range [{lo}, {hi}], got {actual_lo, actual_hi}')
        return ds

    def expand(
        self,
        pcoll: beam.PCollection,
    ) -> beam.PCollection:
        return pcoll | beam.Map(self._validate, expected_dims=self.expected_dims)


fsspec_auth_kwargs = (
    {'headers': {'Authorization': f'Bearer {ED_TOKEN}'}}
    if selected_rel == HTTP_REL
    else {'client_kwargs': earthdata_auth(ED_USERNAME, ED_PASSWORD)}
)
pattern = pattern_from_file_sequence(
    list(gen_data_links(selected_rel)), CONCAT_DIM, fsspec_open_kwargs=fsspec_auth_kwargs
)

remote_and_target_auth_options = {
    'key': os.environ["S3_DEFAULT_AWS_ACCESS_KEY_ID"],
    'secret': os.environ["S3_DEFAULT_AWS_SECRET_ACCESS_KEY"],
    "anon": False,
    'client_kwargs': {
        'region_name': 'us-west-2'
    }
}


def validate_ds(store: zarr.storage.FSStore) -> zarr.storage.FSStore:
    import xarray as xr
    ds = xr.open_dataset(store, engine="zarr", chunks={})
    ds = ds.set_coords(("lat", "lon"))
    #ds = ds.expand_dims(dim="time")
    print(f"[ LEN(TIME) ]: {len(ds['time'])}")
    print(f"[ DS.COORDS ]: {ds.coords}")
    for coord in ["time", "lat", "lon"]:
        assert coord in ds.coords
    return store

recipe = (
    beam.Create(pattern.items())
    | OpenWithKerchunk(
        remote_protocol='s3' if selected_rel == S3_REL else 'https',
        file_type=pattern.file_type,
        storage_options=pattern.fsspec_open_kwargs,
    )
    | WriteCombinedReference(
        store_name=SHORT_NAME,
        concat_dims=pattern.concat_dims,
        identical_dims=IDENTICAL_DIMS,
        precombine_inputs=True,
        target_options=remote_and_target_auth_options,
        remote_options=remote_and_target_auth_options,
        remote_protocol='s3'
    ) | "Validate" >> beam.Map(validate_ds)
)
