
# Purpose: Set up the Docker Network and all containers so they can connect with each other
# Author:  Matt Poulton

############################
####    Docker Prune    ####
############################

docker system prune -a
docker system df

##############################
####    Docker Network    ####
##############################

docker network create gis-dl-local-network
docker network ls
docker network inspect gis-dl-local-network

##############################
####    Docker Compose    ####
##############################

docker-compose -f infrastructure/docker-compose.yml up -d
