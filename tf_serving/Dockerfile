FROM python:3.10.13-slim as base

RUN apt-get update && apt-get clean

RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

