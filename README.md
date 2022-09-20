## What is this?

This is a project to help map addresses to precincts. You might want this if you want
to map a user database with only addresses into precinct subdivisions for the sake of
coordinating volunteer efforts.

## Deploying

This project is designed to be deployed as a docker container. It needs Redis for caching. The easiest way is to use the included docker-compose file.

1. Copy `.env.sample` to `.env`
2. Edit `.env` to put in your Geocoder service API key (google by default). Check https://developers.google.com/maps/documentation/geocoding/get-api-key for instructions on how to get an API key.
3. Launch the containers:

    docker-compose up

You can now test the API by going to http://localhost:8000/docs

## Further customization

The `.env` file serves to pre-fill several required request parameters. All of these are things you can pass with a given request,
but setting them in the .env file means that you can limit the necessary request to just the address or spreadsheet file of addresses.
Remember that your API key is not free, and that if you choose to set the API key in the `.env` file, people using your deployment
may cost you money.

Check out the contents of the docker-compose.yml file. In particular, you may want to change:

- port mapping (default is local port 8000 mapped to port 80 in the container)
