from copyreg import pickle
from tempfile import TemporaryDirectory
from typing import List, Optional
import asyncio
from enum import Enum
from io import BytesIO, StringIO
import os
import pickle

from fastapi import FastAPI, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import geocoder
import pandas
import geopandas as gpd
import requests

import aioredis


REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = os.environ.get("REDIS_PORT", 6379)
REDIS_USER = os.environ.get("REDIS_USER", "redis")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

from logging.config import dictConfig
import logging

from . import logging_conf

dictConfig(logging_conf.LogConfig().dict())
logger = logging.getLogger("geocoder")

app = FastAPI(debug=True)

loop = asyncio.get_event_loop()

@app.get("/")
def read_root():
    return "See the /docs endpoint for the API"

pandas.set_option('display.max_colwidth', None)
pandas.set_option('display.max_columns', None)

class ResolvedAddress(BaseModel):
    address: str
    resolved_address: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    precinct: Optional[int] 
    original_index: Optional[int]

class MapService(str, Enum):
    mapquest = "mapquest"
    google = "google"
    bing = "bing"
    mapbox = "mapbox"
    openstreetmap = "openstreetmap"

def get_redis_cache():
    return aioredis.from_url(
        f"redis://{REDIS_HOST}:{REDIS_PORT}", password=REDIS_PASSWORD,
    )

async def _geocode(address: str, coder_service: MapService, coder_key: str, session: requests.Session, original_index:int = 0):
    cache = get_redis_cache()
    if await cache.exists(address):
        logger.debug("Cache hit for address: {}".format(address))
        return pickle.loads(await cache.get(address))
    logger.info("Cache miss: Geocoding address: {}".format(address))
    result = getattr(geocoder, coder_service)(address, key=coder_key, session=session)
    if result.ok:
        await cache.set(address, pickle.dumps(result.latlng))
    else:
        logger.warning("Geocoder failed to resolve address: {}".format(address))
    return result.latlng[0], result.latlng[1]

async def load_precinct_map(precinct_map_url):
    cache = get_redis_cache()
    if await cache.exists(precinct_map_url):
        logger.debug("Cache hit for precinct map: {}".format(precinct_map_url))
        precinct_map = pickle.loads(await cache.get(precinct_map_url))
    else:
        logger.info("Cache miss: Loading precinct map: {}".format(precinct_map_url))
        precinct_map = gpd.read_file(precinct_map_url)
        await cache.set(precinct_map_url, pickle.dumps(precinct_map))
    return precinct_map

def df_to_return_struct(df, precinct_field_name):
    return [ResolvedAddress(address=row.address, 
                            resolved_address=row.resolved_address, 
                            lat=float(row.lat) if row["lat"] else None, 
                            lng=float(row.lng) if row["lng"] else None, 
                            precinct=int(row[precinct_field_name]) if row[precinct_field_name] else None,
                            original_index=row.original_index)
            for _, row in df.iterrows() if row["address"]]

async def resolve_precinct(resolved_addresses: pandas.DataFrame, precinct_map_url: HttpUrl, precinct_field_name: str):
    gdf = gpd.GeoDataFrame(resolved_addresses, geometry=gpd.points_from_xy(resolved_addresses.lng, resolved_addresses.lat, crs="EPSG:4326"))

    logger.info("Resolving precincts for {} addresses".format(len(gdf)))
    precinct_map = await load_precinct_map(precinct_map_url)
    logger.debug("Precinct map columns: \n{}".format(precinct_map.columns))
    column_subset = list(resolved_addresses.columns) + [precinct_field_name]
    try:
        joined_df = gpd.sjoin(gdf, precinct_map.to_crs("EPSG:4326"), how="left", op="within")[column_subset]
    except KeyError:
        raise HTTPException(status_code=400, detail="Precinct field name not found in precinct map shapefile. Available fields are: {}".format(precinct_map.columns))
    resolved_addresses["precinct"] = joined_df[precinct_field_name]
    resolved_addresses.drop(columns=["geometry", "lat", "lng", "internal_composite_address"], inplace=True)
    # no return; inplace update

@app.get("/geocode", response_model=ResolvedAddress)
async def geocode(address: str, coder_service: Optional[MapService] = MapService.google, coder_key: Optional[str] = None, precinct_map_url: Optional[HttpUrl] = None, precinct_field_name: Optional[str] = None):
    coder_service = coder_service or MapService.google
    coder_key = coder_key or os.environ.get("GEOCODER_KEY")
    if not coder_key:
        raise HTTPException(status_code=400, detail="No geocoder key provided")
    precinct_map_url = precinct_map_url or os.environ.get("PRECINCT_MAP_URL")
    if not precinct_map_url:
        raise HTTPException(status_code=400, detail="No precinct map URL provided")
    precinct_field_name = precinct_field_name or os.environ.get("PRECINCT_FIELD_NAME")
    if not precinct_field_name:
        raise HTTPException(status_code=400, detail="No precinct field name provided")
    with requests.Session() as session:
        latlng = await _geocode(address, coder_service, coder_key, session, original_index=1)
    df = pandas.DataFrame([{"address": address, "lat": latlng[0], "lng": latlng[1], "original_index": 1}])
    # in-place modification of df
    await resolve_precinct(df, precinct_map_url, precinct_field_name)
    return (df_to_return_struct(df, precinct_field_name))[0]

def compose_address_column_and_filter_empty(df, address_columns_str):
    columns = address_columns_str.split(" ")
    only_entries_with_address = df.dropna(subset=columns, how="all")
    df_columns = df.columns
    only_entries_with_address["internal_composite_address"] = df[columns].apply(lambda x: " ".join(x.map(str)), axis=1)
    only_entries_with_address.drop(columns=[col for col in df_columns], inplace=True)
    # this is a collection of addresses that retains the original index numeric values
    return only_entries_with_address

async def read_file(addresses_upload_file):
    first_char = (await addresses_upload_file.read(1)).decode("utf-8")
    await addresses_upload_file.seek(0)
    logger.debug("First character of file: {}".format(first_char))
    if first_char in '{[':
        df = pandas.read_json(addresses_upload_file.file, dtype=str)
        file_type = 'json'
    try:
        df = pandas.read_csv(
                StringIO(str(addresses_upload_file.file.read(), 'utf-8')), encoding='utf-8', dtype=str)
        file_type = 'csv'
    except (UnicodeDecodeError, ValueError) as e:
        logger.debug("Error reading CSV file: {}".format(e))
        logger.debug("Trying to read as Excel file")
        # rewind because we're at the end of the file here
        await addresses_upload_file.seek(0)
        try:
            df = pandas.read_excel(BytesIO(addresses_upload_file.file.read()), engine='openpyxl', dtype=str)
            file_type = 'excel'
        except ValueError as e:
            logger.debug("Error reading Excel file: {}".format(e))
            raise HTTPException(status_code=400, detail="Input file must be CSV, JSON, or Excel")
    return df, file_type

async def _geocode_many(addresses: UploadFile, coder_service: Optional[MapService] = MapService.google, coder_key: Optional[str] = None, precinct_map_url: Optional[HttpUrl] = None, precinct_field_name: Optional[str] = None, address_field_name: Optional[str] = None):
    coder_service = coder_service or os.environ.get("GEOCODER_PROVIDER") or MapService.google
    coder_key = coder_key or os.environ.get("GEOCODER_KEY")
    if not coder_key:
        raise HTTPException(status_code=400, detail="No geocoder key provided")
    precinct_map_url = precinct_map_url or os.environ.get("PRECINCT_MAP_URL")
    if not precinct_map_url:
        raise HTTPException(status_code=400, detail="No precinct map URL provided")
    precinct_field_name = precinct_field_name or os.environ.get("PRECINCT_FIELD_NAME")
    if not precinct_field_name:
        raise HTTPException(status_code=400, detail="No precinct field name provided")
    address_field_name = address_field_name or os.environ.get("ADDRESS_FIELD_NAME") or "address"

    df, file_type = await read_file(addresses)
    if not all([(fn in df.columns) for fn in address_field_name.split()]):
        if address_field_name == "address":
            raise HTTPException(status_code=400, detail="Input file must contain an 'address' column, or you must specify the address_field_name parameter")
        else:
            raise HTTPException(status_code=400, detail="Specified address_field_name(s) '{}' not found in input file".format(address_field_name))
    filtered_df = compose_address_column_and_filter_empty(df, address_field_name)
    with requests.Session() as session:
        filtered_df[["lat", "lng"]] = await asyncio.gather(*[_geocode(address, coder_service=coder_service, coder_key=coder_key, session=session) 
            for address in filtered_df.internal_composite_address])
    await resolve_precinct(filtered_df, precinct_map_url, precinct_field_name) 
    logger.debug("Filtered df head: {}".format(filtered_df.head()))
    # join the filtered data that has addresses with the removed data that doesn't to restore
    # the original table
    merged_df = df.join(filtered_df, how="left")
    logger.debug("Merged dataframe rows: {}".format(len(merged_df)))
    logger.debug("Merged dataframe head: \n{}".format(merged_df.head()))
    return merged_df, file_type

@app.post("/geocode", response_model=List[ResolvedAddress])
async def geocode_csv(addresses: UploadFile, coder_service: Optional[MapService] = MapService.google, coder_key: Optional[str] = None, precinct_map_url: Optional[HttpUrl] = None, precinct_field_name: Optional[str] = None, address_field_name: Optional[str] = None):
    merged_df = await _geocode_many(addresses, coder_service, coder_key, precinct_map_url, precinct_field_name, address_field_name)
    return df_to_return_struct(merged_df, precinct_field_name)

async def get_temp_dir():
    dir = TemporaryDirectory()
    try:
        yield dir.name
    finally:
        del dir

@app.post("/augment", response_class=FileResponse)
async def geocode_augment_csv(addresses: UploadFile, coder_service: Optional[MapService] = MapService.google, coder_key: Optional[str] = None, precinct_map_url: Optional[HttpUrl] = None, precinct_field_name: Optional[str] = None, address_field_name: Optional[str] = None, _dir=Depends(get_temp_dir)):
    merged_df, file_type = await _geocode_many(addresses, coder_service, coder_key, precinct_map_url, precinct_field_name, address_field_name)
    fn = os.path.splitext(addresses.filename)[0] + "-plus_precincts.csv"
    fpath = os.path.join(_dir, fn)
    #if file_type == "csv":
    merged_df.to_csv(fpath, index=False)
    return FileResponse(fpath, filename=fn)
    # TODO: support Excel export. Right now, the files being served are not valid xlsx files and I'm not sure why.
    # elif file_type == "json":
    #     merged_df.to_json(fpath, index=False)
    #     return FileResponse(fpath, filename=fn)
    # elif file_type == "excel":
    #     options = {
    #         "strings_to_urls": False, 
    #         "strings_to_formulas": False,
    #         "nan_inf_to_errors": True,
    #         }
    #     writer = pandas.ExcelWriter(fpath,engine='xlsxwriter',options=options)
    #     merged_df.to_excel(writer, index=False)
    #     writer.save()
    #     return FileResponse(fpath, filename=fn)
    # else:
    #     raise HTTPException(status_code=400, detail="unknown file type, can't augment this")
    
    
    
