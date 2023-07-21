FROM ubuntu:22.04

LABEL author="Adityasaini10012001@gmail.com"

# Lets name it
LABEL purpose="gprMax_Docker_"

# Sync Host and Docker
# ADD . /work

# Specify the work Directory
WORKDIR /home

# Logistics
RUN apt update && apt install -y apt-utils
RUN apt upgrade -y
RUN apt install -y wget gcc g++ git

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN apt install -y bzip2
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="~/miniconda/bin:${PATH}"

RUN git clone --branch devel https://github.com/SainiAditya1/gprMax.git && \
    cd gprMax && \
    git checkout devel &&\
    chmod +x /home/gprMax/Docker/container_init_linux_64.sh && \
    bash /home/gprMax/Docker/container_init_linux_64.sh
