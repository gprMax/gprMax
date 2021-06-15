Setting up gprMax on WSL2
=========================

In this doc, we will go through the process of setting up gprMax on `Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_.

Prerequisites
-------------

Before we build and install gprMax on WSL, the following conditions must be met:

- The Windows OS build version must be **20145 or higher**.
- You must have **WSL2** set up on your Windows machine.
- **NVIDIA preview driver** for WSL2 must be installed.

Updating Windows OS build
^^^^^^^^^^^^^^^^^^^^^^^^^

To check your OS Build Version, you can type ``winver`` in the command prompt.

To get a build of 20145 or higher, you will need to register for the `Windows Insider Program <https://insider.windows.com/>`_. Follow the steps given below to register for the program:

1. Go to *Settings -> Update & Security -> Windows Insider Program*.
2. If you don't see an Error message move to step 4 else follow along.
3. In the Tab, click on the link given below and tick **Optional Diagnostics Data**.
4. Back in Update & Security, click on *Get Started* and complete the registrations.
5. Keep your Insider Settings as **Dev Channel**.
6. After Step 5, check for updates and update your OS.

Enabling WSL2
^^^^^^^^^^^^^

1. Open Command Prompt as Administrator.
2. Type ``wsl --install``.
3. Restart your PC.
4. Install an Ubuntu Distribution from the `Microsoft Store <https://www.microsoft.com/en-us/search?q=ubuntu>`_. (The one used for this demonstration is `Ubuntu 18.04 LTS <https://www.microsoft.com/en-us/p/ubuntu-1804-lts/9n9tngvndl3q?activetab=pivot:overviewtab>`_)

Installing NVIDIA Driver for WSL2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the Driver from `CUDA on WSL <https://developer.nvidia.com/cuda/wsl>`_ page.

**Note:** You will have to register for the *NVIDIA Developer Program* to download this driver.

Once the drivers are downloaded and installed, follow the steps mentioned `here <https://docs.nvidia.com/cuda/wsl-user-guide/index.html#running-cuda>`_. Don't forget to add ``sudo`` before the commands given in the doc previously mentioned.

**Note:** Check for the relevant `meta-packages <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-metas>`_ and the **latest version** of the toolkit.

There is no need to setup the Containers for our purpose.

After this, open the ``.bashrc`` file and add the following line to it:

.. code-block:: bash

    export PATH="$PATH:/wherever/cuda/bin/is/located"

This will add the commands to PATH. Restart the Ubuntu terminal after doing so and check if the ``nvcc`` Compilation tools have the same version as the Driver. You can check that using the following command;

.. code-block:: bash

    $ nvcc --version

Installation of gprMax
----------------------

You will need to install some libraries manually before you can create a gprMax's virtual environment. You will also need to download and install Miniconda. Here's a `helpful guide <https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da>`_ on how to do that.

After installing Miniconda, install Open MPI using the commands:

.. code-block:: bash

    $ sudo apt install libopenmpi-dev
    $ pip install mpi4py

Then you need to install PyCUDA manually. Get the latest release from `here <https://pypi.org/project/pycuda/2021.1/#history>`_ and follow `this guide <https://wiki.tiker.net/PyCuda/Installation/Linux/>`_ to install PyCUDA on WSL.

**Note:** Install ``pytest`` using ``pip install pytest`` before running the PyCUDA test. You may also see warning messages during the testing of PyCUDA.

After everything is done, follow the normal procedure to create the virtual environment and building and installing gprMax.
