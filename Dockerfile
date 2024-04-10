FROM ubuntu:22.04

WORKDIR /IL-Datasets
COPY . /IL-Datasets

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y xdg-utils

RUN pip install torch torchvision 
RUN pip install gymnasium[box2d] gym
RUN pip install -e .[benchmark]

RUN echo 'alias python="python3"' >> ~/.bashrc

# Demo related
RUN pip install notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
