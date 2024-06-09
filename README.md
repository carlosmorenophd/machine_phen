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
