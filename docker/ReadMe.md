# gprMax Dockerfile

This repository contains a Dockerfile that creates a [Docker](https://www.docker.com/) image with [gprMax](http://www.gprmax.com/index.shtml) and its dependencies installed via [miniconda](https://docs.conda.io/en/latest/miniconda.html). This includes installed/built official repository of gprMax.

If you still have not done so, [install Docker](https://docs.docker.com/engine/getstarted/step_one/) and [verify your installation](https://docs.docker.com/engine/getstarted/step_three/).

Pull docker image
-----------

You can download (pull) the image via following command:

     $ docker pull nipruthi/gprmax


Run docker image
-----------

Run:

    $ docker run -it nipruthi/gprmax


### Run with access to a folder in the host machine

Maybe you want to include a local folder with GNSS raw data to process in your container, or to extract output files generated during its execution. You can do that by running the container as:

    $ docker run -it -v /home/user/data:/data nipruthi/gprmax

This will mount the `/home/user/data` folder in the host machine on the `/data` folder inside the container, with read and write permissions.


### Run with graphical environment:

 * **On GNU/Linux host machines with X11 server installed**

   In the host machine, adjust the permission of the X server host by the following command:

       $ xhost +local:root

   Then run the container with:

       $ docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY \
         nipruthi/gprmax:v1 /bin/bash

   In case you want to revoke the granted permission:

       $ xhost -local:root

 * **Test it!**

   In the container:
   
       
       $ root@ubuntu:/home#           conda activate gprMax
       $ root@ubuntu:/home#           cd gprMax
       $ root@ubuntu:/home/gprMax#    python -m gprMax user_models/cylinder_Ascan_2D.in
       $ root@ubuntu:/home/gprMax#    python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out
       



Build docker image
-----------

This step is not needed if you have pulled the docker image. If you want to build the Docker image on you own, go to the repository docker folder and run the following command:

     $ docker build -t nipruthi/gprmax:v1 .

You can change `nipruthi/gprmax:v1` at your own preference.
