# GSoC 2022 - Project Ideas List

gprMax is planning to participate in the [Google Summer of Code](https://summerofcode.withgoogle.com) 2022 program, following a successful participation in 2019 and 2021. Here is list of some potential project ideas (in no particular order):

## 1. Multi-GPU model execution

The aim of the project is to investigate multi-GPU model execution, i.e. allow a model to execute (and share memory) across multiple GPUs.

Currently with our GPU-based (PyCuda) solver, a model must fit within the memory of a single GPU. Simulations are becoming ever larger and more complex, which often means their memory requirements exceed that available on a single GPU. A solution is required to allow a model to execute (and share memory) across multiple GPUs. This may involve a MPI type domain decomposition or simpler memory sharing approach.

**Skills required:** Python, CUDA.

**Difficulty:** Hard

**Length:** 350hrs

**Mentor(s):** Dr Craig Warren (craig@gprmax.com) and Dr Antonis Giannopoulos (antonis@gprmax.com) 


## 2. Web-based framework for model building

The aim of this project is to develop a web-based framework that allows models to be graphically built.

Many models, especially for Ground Penetrating Radar (GPR), can be easily specified using a text-based input file, which is currently what is done. This approach can be beneficial when executing large simulations in [high-performance computing (HPC)](https://en.wikipedia.org/wiki/Supercomputer) environments. However, there are also simulations that require fine, complex details to be modelled or where existing geometries already exist (for example in CAD format). In these cases a graphical-based model building environment would be beneficial, and one that does not require bespoke software to be installed. A web-based framework would therefore be a very useful for model building and construction.

**Skills required:** Python, web frameworks for 2D/3D visualisation.

**Difficulty:** Medium

**Length:** 350hrs

**Mentor(s):** Dr John Hartley (johnmatthewhartley@gmail.com) and Dimitris Angelis ()


## 3. Independent model building and execution

The aim of this project is to investigate separating model building and execution phases.

Currently the modeling building phase takes place (which is mostly a serial process) on CPU. Once that is complete the model can then be executed using either the CPU-based (OpenMP) or GPU-based (CUDA) solver. The next model cannot be built until the previous model has finished executing, and this can sometimes cause a performance bottleneck. If the model building and execution phases could be made more independent and a queuing system implemented then this may solve this performance problem.

**Skills required:** Python, CUDA.

**Difficulty:** Medium

**Length:** 350hrs

**Mentor(s):** Dr Craig Warren (craig@gprmax.com)


## 4. Optimising GPU performance

The aim of the project is to optimise our CUDA-based solver for GPUs. The performance (speed) of the solver is a critical feature as simulations become ever larger and more complex.

The solver is based on the [Finite-Difference Time-Domain (FDTD)](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) method, which has shown significant performance benefits when parallelised – particularly on GPU. We have GPU-based solver which uses NVIDIA CUDA (with PyCUDA). The speed-up is significant (x30 compared to parallelised CPU), but we would like to further tune and optimise the performance. 

**Skills required:** Python, CUDA.

**Difficulty:** Medium

**Length:** 175hrs

**Mentor(s):** Dr Craig Warren (craig@gprmax.com)


## 5. Improved installation tools

The aim of this project is to create a simplified and more user-friendly installation workflow for the software.

gprMax is predominately written in Python, but some of the performance-critical parts of the code are written in [Cython](https://cython.org), which must be built and compiled. The current installation involves building a Python environment, installing a C compiler with OpenMP support, and building and installing the gprMax package. This can be a lengthy and complex procedure, depending on your operating system, especially for first-time or inexperienced users.

**Skills required:** Python, Cython, tools and compilers on multiple (Linux, Windows, macOS) operating systems

**Difficulty:** Medium

**Length:** 175hrs

**Mentor(s):** Dr John Hartley (johnmatthewhartley@gmail.com)


## 6. Comprehensive test and benchmarking suite

The aim of this project is to develop a comprehensive test suite and benchmarking toolset.

Currently gprMax includes a series of tests that verify specific simulation results against reference solutions. This only tests large chunks of code at a relatively high level. As the functionality and complexity of the code base increases a more comprehensive, fine-grained set of tests is required. The performance of the code is also an important element, as simulations can be extremely large and complex, with some requiring weeks of runtime on [high-performance computing (HPC)](https://en.wikipedia.org/wiki/Supercomputer). It is therefore key to have some models that can be used to benchmark code features with both the multi-threaded CPU (OpenMP), and GPU accelerated solvers.

**Skills required:** Python, tools and compilers on multiple (Linux, Windows, macOS) operating systems, some knowledge of GPU and HPC would be beneficial.

**Difficulty:** Medium

**Length:** 350hrs

**Mentor(s):** Dr Craig Warren (craig@gprmax.com) 


## 7. Web-based geometry visualisation

The aim of this project is create a web-based viewer for the visualisation of model geometries.

Being able to visualise and check the geometry and materials of models before running simulations is vital. It minimises the risk of wasting computational resources by running incorrect models. Currently gprMax uses the [Visualization Toolkit (VTK)](https://vtk.org) format to store geometry information to file, and [Paraview](https://www.paraview.org) for visualisation. This allows the geometrical information to be viewed but does not easily permit material information from the model to be checked. A tighter integration is required, possibly considering a browser-based solution within Jupyter notebooks.

**Skills required:** Python, Jupyter notebooks, familiarity with VTK and Paraview would beneficial.

**Difficulty:** Medium

**Length:** 175hrs

**Mentor(s):** Dr John Hartley (johnmatthewhartley@gmail.com)


## 8. Arbitary placement of Perfectly Matched Layer (PML) material

The aim of this project is to provide the ability for users to use Perfectly Matched Layer (PML) material with geometry primitives, i.e. PML material should not be confined to being used at the boundaries of the domain.

Currently PMLs are used as absorbing boundaries to effectively and efficiently terminate the model domain. There are scenarios where use of the PML material with geometry primitives, i.e. boxes, cylinders etc..., would be beneficial, e.g. building antenna models.

**Skills required:** Python.

**Difficulty:** Medium

**Length:** 175hrs

**Mentor(s):** Dr Antonis Giannopoulos (antonis@gprmax.com)
