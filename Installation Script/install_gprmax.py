
import os
import sys
import platform
import subprocess

import urllib.request

import shutil

def is_conda_installed():
    try:
        subprocess.check_output(['conda', '--version'])
        return True
    except FileNotFoundError:
        return False
    
    
def install_conda():
    if platform.system() == "Windows":
        miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
        os.system(f"powershell -Command \"Invoke-WebRequest '{miniconda_url}' -OutFile 'Miniconda3-latest-Windows-x86_64.exe'\"")
        os.system("start Miniconda3-latest-Windows-x86_64.exe")
        input("Press Enter when Miniconda installation is complete...")
        
        # Add Conda folders to PATH
        conda_folders = [
            os.path.join(os.path.expanduser("~"), "miniconda3", "Scripts"),
            os.path.join(os.path.expanduser("~"), "miniconda3", "Library", "bin")
        ]
        current_path = os.environ.get("PATH", "")
        new_path = os.pathsep.join(conda_folders + [current_path])
        os.environ["PATH"] = new_path

    elif platform.system() == "Linux":
        miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        os.system(f"wget {miniconda_url}")
        os.system("chmod +x Miniconda3-latest-Linux-x86_64.sh")
        os.system("./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3")
        
        # Add Conda folders to PATH
        conda_folders = [
            os.path.join(os.path.expanduser("~"), "miniconda3", "bin")
        ]
        current_path = os.environ.get("PATH", "")
        new_path = os.pathsep.join(conda_folders + [current_path])
        os.environ["PATH"] = new_path

    elif platform.system() == "Darwin":
        if platform.machine().endswith("64"):
            miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        elif platform.machine().endswith("arm64"):
            miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        os.system(f"curl -O {miniconda_url}")
        os.system("chmod +x " + os.path.basename(miniconda_url))
        os.system(f"./{os.path.basename(miniconda_url)} -b -p $HOME/miniconda3")
        
        # Add Conda folders to PATH
        conda_folders = [
            os.path.join(os.path.expanduser("~"), "miniconda3", "bin")
        ]
        current_path = os.environ.get("PATH", "")
        new_path = os.pathsep.join(conda_folders + [current_path])
        os.environ["PATH"] = new_path    
            
        
def is_gprmax_environment_present():
    try:
        output = subprocess.check_output(['conda', 'info', '--envs']).decode('utf-8')
        return 'gprMax' in output
    except subprocess.CalledProcessError:
        return False     
    
def print_options():
   
    print("1. Update gprMax")
    # print("2. Install gprMax at other directory")
    print("2. Abort installation")

def get_option():
    option = input("Enter your option: ")
    return option   



# def choose_directory():
#     while True:
#         directory = input("Enter the directory path: ")
#         if os.path.isdir(directory):
#             return os.path.abspath(directory)
#         else:
#             print("Invalid directory. Please try again.")    
def switch_to_directory():
    # Ask the user for a directory path
    directory_path = input("Enter the directory path: ")

    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # Change the current working directory to the specified directory
        os.chdir(directory_path)
        print(f"Switched to directory: {directory_path}")
    else:
        print("Invalid directory path.")
        exit()
            
            
def activate_conda_environment(environment_name):
    conda_path = subprocess.check_output('conda info --base', shell=True, universal_newlines=True).strip()
    activate_script = os.path.join(conda_path, 'etc', 'profile.d', 'conda.sh')
    
    activate_command = f'source {activate_script} && conda activate {environment_name}'
    subprocess.run(activate_command, shell=True, executable="/bin/bash")  
         
    
    
def update_gprMax():
    
    
    try:
        # Run 'git pull' command to pull changes from the remote repository
        switch_to_directory()
        pull_output = subprocess.check_output(['git', 'pull']).decode('utf-8')
        
        # Check if the output contains 'Already up to date', indicating that no new changes were pulled
        if 'Already up to date' in pull_output:
            print("Already up to date.")
            sys.exit()
        else:
            if platform.system()!="Windows":
                original_directory = os.getcwd()
                # switch_to_directory()
                activate_conda_environment("gprMax-devel")
                subprocess.run("git checkout devel",shell=True)
                subprocess.run("git pull",shell=True)
                subprocess.run(["python3" ,"setup.py", "cleanall"],check=True)
                subprocess.run(["python3", "setup.py", "build"],check=True)
                subprocess.run(["python3", "setup.py", "install"],check=True)
                print("Updated Sucessfully")
                sys.exit()
        
            else:
                # switch_to_directory()
                subprocess.run("conda.bat activate gprMax-devel ", shell = True)
                subprocess.run("git checkout devel",shell=True)
                subprocess.run("git pull",shell=True)
                subprocess.run(["python" ,"setup.py", "cleanall"],check=True)
                subprocess.run(["python", "setup.py", "build"],check=True)
                subprocess.run(["python", "setup.py", "install"],check=True)
                print("Updated Sucessfully")
                sys.exit()
            
    
    except subprocess.CalledProcessError as e:
        print(f"Error updating gprMax: {e}")
    
        
def buildtoolwindow():
    # URL to download Microsoft Build Tools for Visual Studio 2022 (direct link)
    download_url = "https://aka.ms/vs/17/release/vs_buildtools.exe"

    # Define the target directory where the installer will be downloaded
    download_directory = os.path.join(os.path.expanduser("~"), "Downloads")
    installer_file = os.path.join(download_directory, "vs_buildtools.exe")

    # Path to add to the system's Path environment variable
    msvc_bin_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

    def download_build_tools():
        # Create the download directory if it doesn't exist
        if not os.path.exists(download_directory):
            os.makedirs(download_directory)

        # Download the Microsoft Build Tools installer
        print("Downloading Microsoft Build Tools for Visual Studio 2022...")
        try:
            urllib.request.urlretrieve(download_url, installer_file)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download the installer: {e}")
            exit(1)

    def install_build_tools():
        # Run the installer with necessary arguments
        print("Installing Microsoft Build Tools for Visual Studio 2022...")
        try:
            subprocess.run([installer_file, "--quiet", "--norestart", "--wait", "--add", "Microsoft.VisualStudio.Workload.NativeDesktop", "--includeRecommended", "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041", "--add", "Microsoft.VisualStudio.Component.Windows10SDK.22000", "--add", "Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win81", "--add", "Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win10"])
            print("Installation completed.")
        except Exception as e:
            print(f"Failed to install the Build Tools: {e}")
            exit(1)

    def set_path_environment_variable():
        # Add the MSVC bin path to the system's Path environment variable
        try:
            current_path = os.environ.get('Path', '')
            if msvc_bin_path not in current_path:
                updated_path = f"{current_path};{msvc_bin_path}"
                os.environ['Path'] = updated_path
                print("Path environment variable updated.")
            else:
                print("Path environment variable already contains the necessary entry.")
        except Exception as e:
            print(f"Failed to set Path environment variable: {e}")

    # Main function
    
    try:
        download_build_tools()
        install_build_tools()
        set_path_environment_variable()
        print("Build Tools installation completed.")
    except Exception as e:
        print(f"Error during Build Tools Installation installation: {e}")
    
       
def is_git_installed():
    try:
        subprocess.run("git --version", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True  
    except subprocess.CalledProcessError:
            return False  
def install_git_with_conda():
    result = subprocess.run(["conda", "install", "git", "-y"])
    if result.returncode != 0:
        print("Error installing Git. Aborting installation.")
        return False
    return True    
    
        
def install_gprMax():
    print("Welcome to the gprMax installation script!")
    if is_conda_installed():
        print("Conda is installed on the system.")
        if is_gprmax_environment_present():
            print("gprmax environment is already present.")
            print_options()
            option = get_option()
            if option == "1":
                print("Updating gprMax...")
                update_gprMax()
                exit()
            # elif option == "2":
            #     print("Installing gprMax at another directory...")
            #     # to be implemeted
                
                # selected_directory = choose_directory()
                # subprocess.call(['bash', '-c', 'cd "{}" && exec bash'.format(selected_directory)])
            elif option == "2":
                print("Aborting installation...")
                exit()
            else:
                print("Invalid option")
        else:
            print("gprmax environment is not found. Installing gprMax...")
    
    else:
        print("Conda is not installed on the system.")
        continue_install = input("Continue installation? Enter 'yes' to continue: ")
        if continue_install.lower() == "yes":
            install_conda()
        else:
            print("Installation aborted.")
            exit()


    # Step 2: Install Git and clone gprMax repository
    
    
    result = subprocess.run(["conda", "install", "git", "-y"])
    if result.returncode != 0:
        print("Error installing Git. Aborting installation.")
        return
    result = subprocess.run(["git", "clone", "https://github.com/gprMax/gprMax.git", "-b", "devel"])
    if result.returncode != 0:
        print("Error cloning gprMax repository. Aborting installation.")
        return
    
    os.chdir("gprMax")

    continue_install = input("Continue installation? Enter 'yes' to continue: ")
    if continue_install.lower() != "yes":
        print("Installation aborted.")
        return

    # Step 3: Create conda environment and install dependencies
    result = subprocess.run(["conda", "env", "create", "-f", "conda_env.yml"])
    if result.returncode != 0:
        print("Error creating conda environment. Aborting installation.")
        return

    # Step 4: Install C compiler supporting OpenMP (Windows, Ubuntu, and macOS)
    if platform.system() == "Windows":
        buildtoolwindow()

    elif platform.system() == "Linux":

        # gcc should be already installed on Linux, so no action required
        pass

    elif platform.system() == "Darwin":
        # Install gcc on macOS using Homebrew
        result = subprocess.run(["brew", "install", "gcc"])
        if result.returncode !=0:
            print("Error Installing C compiler. Aborting installation.")
            return 

    # Step 5: Build and install gprMax
    os.chdir( "gprMax" )
    if platform.system()!="Windows":
        activate_conda_environment("gprMax-devel")
        subprocess.run(["python3", "setup.py", "build"],check=True)
        subprocess.run(["python3", "setup.py", "install"],check=True)
       
    else:
        subprocess.run("conda.bat activate gprMax-devel " , shell = True)
        subprocess.run(["python", "setup.py", "build"], check=True)
        subprocess.run(["python", "setup.py", "install"], check=True)

    print("gprMax installation complete.")


install_gprMax()

