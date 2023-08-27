
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
