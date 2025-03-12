#This file is used to create a custom gprMax-devel kernel in the Jupyter notebook environment. Running this from the terminal in
#linux after a regular gprMax installation would automatically create the kernel for you and then the models can be run within a
#notebook environment.

import subprocess

def install_packages():
    try:
        subprocess.run(["conda", "install", "-c", "anaconda", "ipykernel"], check=True)
        subprocess.run(["python", "-m", "ipykernel", "install", "--user", "--name=gprMax-devel"], check=True)
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)

if __name__ == "__main__":
    install_packages()
