FROM ubuntu:22.04

WORKDIR /IL-Datasets
COPY . /IL-Datasets

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch torchvision gymnasium
RUN pip install -e .

RUN echo 'alias python="python3"' >> ~/.bashrc
