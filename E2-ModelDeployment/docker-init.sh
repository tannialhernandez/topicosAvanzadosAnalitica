#!/bin/bash
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker build -t taller .
docker run --name taller -p 5000:5000 taller