This is a project to help map addresses to precincts. You might want this if you want
to map a user database with only addresses into precinct subdivisions for the sake of
coordinating volunteer efforts.

This project is designed to be deployed as a docker container. To build a container:

    docker build -t geo .

To run the server:

    docker run -p 8000:80 -it geo:latest

The -p argument maps port 8000 on your computer to port 80 inside the container.

You can now test the API by going to http://localhost:8000/docs

* The coder_key parameter is your API key for a given geocoding service.
* The precinct_map_url is the URL to download the map shapefile from. For Texas, the best one is at https://data.capitol.texas.gov/dataset/d04c72b9-16c4-4ab2-8c6d-c666d41e04b7/resource/fb56da88-63d5-44a9-9577-63d1b654a8ab/download/precincts22p_20220518.zip
* The precinct_field_name is the column name that has the numeric precinct ID's. For the map above, the key is PREC
