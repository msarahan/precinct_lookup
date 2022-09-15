This is a project to help map addresses to precincts. You might want this if you want
to map a user database with only addresses into precinct subdivisions for the sake of
coordinating volunteer efforts.

This project is designed to be deployed as a docker container. It needs Redis for caching. The easiest way is to use the included docker-compose file.

    docker-compose up

You can now test the API by going to http://localhost:8000/docs

Check out the contents of the docker-compose.yml file. In particular, you may want to change:

- port mapping (default is local port 8000 mapped to port 80 in the container)
- environment variables that can greatly simplify your queries:
  - GEOCODER_PROVIDER: default geocoder service
  - GEOCODER_KEY: your API key for your geocoder provider
  - PRECINCT_MAP_URL: URL to download Shapefile for precinct boundaries
  - PRECINCT_FIELD_NAME: field name in the Shapefile for the precinct numbers
  - ADDRESS_FIELD_NAME: When uploading a spreadsheet of some kind, you can specify the field(s) to be used to get the address from the spreadsheet. If you want to have more than one field, separate them with a space. Be careful with quoting here when setting this from a terminal: `ADDRESS_FIELD_NAME="STREET CITY ZIP"` would concatenate data from the STREET, CITY, and ZIP fields.

Side notes:
* The precinct_map_url is the URL to download the map shapefile from. For Texas, the best one is at https://data.capitol.texas.gov/dataset/d04c72b9-16c4-4ab2-8c6d-c666d41e04b7/resource/fb56da88-63d5-44a9-9577-63d1b654a8ab/download/precincts22p_20220518.zip
* The PRECINCT_FIELD_NAME for the Texas dataset above is PREC
