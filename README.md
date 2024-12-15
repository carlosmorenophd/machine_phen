# To run don docker

## Run on production
Command to build image

```
docker build --tag phen/machine:24.06 .
```

To run the container

```
docker run --name machine -p 8082:8000 -d --network=net-phenotypic -e DATABASE_USERNAME=admin -e DATABASE_PASSWORD=Fantasy24 -e DATABASE=phenotypic_db -e DATABASE_HOST=mariadb_phenotypic -e DATABASE_SOCKET=3306 -e FTP_HOSTNAME=ftp_phenotypic -e FTP_PORT=21 -e FTP_USERNAME=user -e FTP_PASSWORD=ftp1221Wheat phen/machine:24.06
```

## Run on dev

Create image to dev project

```
docker build --tag phen/machine:00.dev -f Dockerfile.dev .
```

Run image in a container

```
docker run -it -d --name dev_machine --network=net-phenotypic -v ${PWD}:/develop  phen/machine:00.dev
```

# New documentation
Run contained on dev

Build the images
```
docker compose -f compose.dev.yaml build
```

Run the contained
```
docker compose -f compose.dev.yaml up -d
```
Access to docker
`docker compose -f compose.dev.yaml exec -it tasksdd bash`

To run the celery task

```
watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery -A tasks worker --loglevel=INFO
```

Run to some test

selection variable by force brute 

```
python tasks_test.py regression_forward_force_single_machine_single_file lrace_trueba_fill_clean_normalize.csv Rendimiento '{"name": "random_forest_regression"}' 

```

selection variable by genetic algorithm

```
python tasks_test.py regression_genetic_single_machine_single_file lrace_trueba_fill_clean_normalize.csv Rendimiento '{"name": "random_forest_regression"}' '{"num_generations":100}'

```

```
python tasks_test.py regression_genetic_single_machine_single_file lrace_all_clean_fill_normalizel.csv GrainYield '{"name": "random_forest_regression"}' '{"num_generations":100}'
```


