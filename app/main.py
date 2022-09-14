from typing import List, Optional
import asyncio
from enum import Enum

from aiocache import caches
from fastapi import FastAPI, HTTPException
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
                            precinct=int(row[precinct_field_name]) or 0)
            for _, row in joined_df.iterrows()]

@app.get("/geocode", response_model=ResolvedAddress)
async def geocode(address: str, coder_service: MapService, coder_key: str, precinct_map_url: HttpUrl, precinct_field_name: str):
    with requests.Session() as session:
        ra = await _geocode(address, coder_service, coder_key, session)
    return (await resolve_precinct([ra], precinct_map_url, precinct_field_name))[0]

@app.post("/geocode", response_model=List[ResolvedAddress])
async def geocode_csv(addresses: str, coder_service: MapService, coder_key: str, precinct_map_url: HttpUrl, precinct_field_name: str):
    with open('addresses.csv', 'w') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char in '{[':
            df = pandas.read_json(f)
        try:
            df = pandas.read_csv(f)
        except ValueError:
            try:
                df = pandas.read_excel(f)
            except ValueError:
                raise HTTPException(status_code=400, detail="Input file must be CSV, JSON, or Excel")
    
    if 'address' not in df.columns:
        raise HTTPException(status_code=400, detail="No 'address' column in input file")
    with requests.Session() as session:
        resolved_addresses = await asyncio.gather(*[_geocode(address, coder_service=coder_service, coder_key=coder_key, session=session) for address in df['address']])
    return await resolve_precinct(resolved_addresses, precinct_map_url, precinct_field_name)
