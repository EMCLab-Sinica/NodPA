# docker build -f docker/Dockerfile -t yan12125/nodpa .
# docker run --rm -it -v "${PWD}:/NodPA" yan12125/nodpa

FROM ubuntu:22.04

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get -y install --no-install-recommends --no-install-suggests python3-pip cmake make g++

RUN python3 -m pip install --upgrade pip setuptools

COPY requirements-base.txt .

RUN python3 -m pip install -r requirements-base.txt
