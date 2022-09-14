from typing import List, Optional
import asyncio
from enum import Enum
from io import BytesIO, StringIO

from aiocache import caches
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, HttpUrl
import geocoder
import pandas
import geopandas as gpd
import requests

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


caches.set_config({
    'default': {
        'cache': "aiocache.SimpleMemoryCache",
        'serializer': {
            'class': "aiocache.serializers.NullSerializer"
        }
    }
})

pandas.set_option('display.max_colwidth', None)
pandas.set_option('display.max_columns', None)

class ResolvedAddress(BaseModel):
    address: str
    resolved_address: Optional[str]
    lat: Optional[float]
    lng: Optional[float]
    precinct: Optional[int] 

class MapService(str, Enum):
    mapquest = "mapquest"
    google = "google"
    bing = "bing"
    mapbox = "mapbox"
    openstreetmap = "openstreetmap"

async def _geocode(address: str, coder_service: MapService, coder_key: str, session: requests.Session):
    cache = caches.get('default')
    if await cache.exists(address):
        logger.debug("Cache hit for address: {}".format(address))
        return await cache.get(address)
    ra = ResolvedAddress(address=address)
    logger.info("Cache miss: Geocoding address: {}".format(address))
    result = getattr(geocoder, coder_service)(address, key=coder_key, session=session)
    if result.ok:
        ra.resolved_address = result.address
        ra.lat, ra.lng = result.latlng
        await cache.set(address, ra)
    else:
        logger.warning("Geocoder failed to resolve address: {}".format(address))
    return ra

async def load_precinct_map(precinct_map_url):
    cache = caches.get('default')
    if await cache.exists(precinct_map_url):
        logger.debug("Cache hit for precinct map: {}".format(precinct_map_url))
        precinct_map = await cache.get(precinct_map_url)
    else:
        logger.info("Cache miss: Loading precinct map: {}".format(precinct_map_url))
        precinct_map = gpd.read_file(precinct_map_url)
        await cache.set(precinct_map_url, precinct_map)
    return precinct_map

async def resolve_precinct(resolved_addresses: List[ResolvedAddress], precinct_map_url: HttpUrl, precinct_field_name: str):
    df = pandas.DataFrame([ra.dict() for ra in resolved_addresses])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat, crs="EPSG:4326"))

    logger.info("Resolving precincts for {} addresses".format(len(gdf)))
    precinct_map = await load_precinct_map(precinct_map_url)
    logger.debug("Precinct map columns: \n{}".format(precinct_map.columns))
    column_subset = list(df.columns) + [precinct_field_name]
    try:
        joined_df = gpd.sjoin(gdf, precinct_map.to_crs("EPSG:4326"), how="left", op="within")[column_subset]
        logger.debug("Joined dataframe head: \n{}".format(joined_df.head()))
    except KeyError:
        raise HTTPException(status_code=400, detail="Precinct field name not found in precinct map shapefile. Available fields are: {}".format(precinct_map.columns))
    return [ResolvedAddress(address=row.address, 
                            resolved_address=row.resolved_address, 
                            lat=row.lat, 
                            lng=row.lng, 
                            precinct=0 if pandas.isna(row[precinct_field_name]) else int(row[precinct_field_name]))
            for _, row in joined_df.iterrows()]

@app.get("/geocode", response_model=ResolvedAddress)
async def geocode(address: str, coder_service: MapService, coder_key: str, precinct_map_url: HttpUrl, precinct_field_name: str):
    with requests.Session() as session:
        ra = await _geocode(address, coder_service, coder_key, session)
    return (await resolve_precinct([ra], precinct_map_url, precinct_field_name))[0]

@app.post("/geocode", response_model=List[ResolvedAddress])
async def geocode_csv(addresses: UploadFile, coder_service: MapService, coder_key: str, precinct_map_url: HttpUrl, precinct_field_name: str, address_field_name: str = "address"):
    first_char = (await addresses.read(1)).decode("utf-8")
    await addresses.seek(0)
    logger.debug("First character of file: {}".format(first_char))
    if first_char in '{[':
        df = pandas.read_json(addresses.file)
    try:
        df = pandas.read_csv(
                StringIO(str(addresses.file.read(), 'utf-8')), encoding='utf-8')
    except (UnicodeDecodeError, ValueError) as e:
        logger.debug("Error reading CSV file: {}".format(e))
        logger.debug("Trying to read as Excel file")
        # rewind because we're at the end of the file here
        await addresses.seek(0)
        try:
            df = pandas.read_excel(BytesIO(addresses.file.read()), engine='openpyxl')
        except ValueError as e:
            logger.debug("Error reading Excel file: {}".format(e))
            raise HTTPException(status_code=400, detail="Input file must be CSV, JSON, or Excel")

    if not all([(fn in df.columns) for fn in address_field_name.split()]):
        if address_field_name == "address":
            raise HTTPException(status_code=400, detail="Input file must contain an 'address' column, or you must specify the address_field_name parameter")
        else:
            raise HTTPException(status_code=400, detail="Specified address_field_name(s) '{}' not found in input file".format(address_field_name))
    with requests.Session() as session:
        resolved_addresses = await asyncio.gather(*[_geocode(address, coder_service=coder_service, coder_key=coder_key, session=session) for address in df[address_field_name.split()].apply(lambda x: " ".join(x.map(str)), axis=1)])
    return await resolve_precinct(resolved_addresses, precinct_map_url, precinct_field_name)
