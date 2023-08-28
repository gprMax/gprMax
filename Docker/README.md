# Steps to build the Docker Image :
## Prerequisites:
 ### Before you can run gprMax, ensure you have Docker installed on your system. Follow these steps to get Docker up and running:
 ### 1.Install Docker:
 ###   If you don't have Docker installed, you can download and install it from the official Docker website: [Docker Installation Guide](https://docs.docker.com/get-docker/)
 ### 2.Verify Installation:
 ### After installing Docker, verify that it's properly installed by running the following command in your terminal:
 ```
docker --version
```
### You should see the version of Docker you just installed printed in the terminal.
### Now You are ready to build the Image
```bash
cd gprMax
```
```bash
cd Docker
```
## You can build the docker image by using the following command
``` bash
sudo docker build -t gprmax_latest1.0
```
## After building the docker image you can run it by the following command
```bash
sudo docker run -it  gprmax_latest1.0
```
## After running the Docker Image ,You can run one of the test models:
## In the container:
```
       $ root@ubuntu:/home#           conda activate gprMax-devel
       $ root@ubuntu:/home#           cd gprMax
       $ root@ubuntu:/home/gprMax#    python -m gprMax examples/cylinder_Ascan_2D.in
       $ root@ubuntu:/home/gprMax#    python -m toolboxes.Plotting.plot_Ascan examples/cylinder_Ascan_2D.h5
```
### Your results should be like those from the A-scan from the metal cylinder example in:
### [introductory/basic 2D models section](https://docs.gprmax.com/en/latest/examples_simple_2D.html#view-the-results)
## Basic  Usage of gprMax is:
```
       $ root@ubuntu:/home/gprMax#      python -m gprMax path_to/name_of_input_file
```
