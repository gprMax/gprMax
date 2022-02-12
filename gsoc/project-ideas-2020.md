# GSoC 2020 - Project Ideas List

gprMax is planning to participate in the [Google Summer of Code](https://summerofcode.withgoogle.com) 2020 program, following a successful debut last year. Here is list of some potential project ideas (in no particular order):


## 1. MPI domain decomposition

The aim of the project is to create version of the solver engine that uses domain decomposition based on the [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) model.

Simulations are becoming ever larger and more complex, which often means their memory requirements exceed that available on a single machine or [high-performance computing (HPC)](https://en.wikipedia.org/wiki/Supercomputer) node. It is possible to use MPI to decompose (or split up) the model domain, and each part can be computed on separate machines or nodes of a HPC.

**Skills required:** Python, C, MPI, some knowledge of HPC environments would be beneficial.

**Difficulty:** Hard

**Mentor(s):** Dr Antonis Giannopoulos (antonis@gprmax.com)


## 2. Improved installation tools

The aim of this project is to create a simplified and more user-friendly installation workflow for the software.

gprMax is predominately written in Python, but some of the performance-critical parts of the code are written in [Cython](https://cython.org), which must be built and compiled. The current installation involves building a Python environment, installing a C compiler with OpenMP support, and building and installing the gprMax package. This can be a lengthy and complex procedure, depending on your operating system, especially for first-time or inexperienced users.

**Skills required:** Python, Cython, tools and compilers on multiple (Linux, Windows, macOS) operating systems

**Difficulty:** Medium

**Mentor(s):** Dr Craig Warren (craig@gprmax.com)


## 3. Comprehensive test and benchmarking suite

The aim of this project is to develop a comprehensive test suite and benchmarking toolset.

Currently gprMax includes a series of tests that verify specific simulation results against reference solutions. This only tests large chunks of code at a relatively high level. As the functionality and complexity of the code base increases a more comprehensive, fine-grained set of tests is required. The performance of the code is also an important element, as simulations can be extremely large and complex, with some requiring weeks of runtime on [high-performance computing (HPC)](https://en.wikipedia.org/wiki/Supercomputer). It is therefore key to have some models that can be used to benchmark code features with both the multi-threaded CPU (OpenMP), and GPU accelerated solvers.

**Skills required:** Python, tools and compilers on multiple (Linux, Windows, macOS) operating systems, some knowledge of GPU and HPC would be beneficial.

**Difficulty:** Medium

**Mentor(s):** Dr Antonis Giannopoulos (antonis@gprmax.com) and Dr Craig Warren (craig@gprmax.com)


## 4. Geometry visualisation

The aim of this project is to improve the handling and visualisation of model geometries.

Being able to visualise and check the geometry and materials of models before running simulations is vital. It minimises the risk of wasting computational resources by running incorrect models. Currently gprMax uses the [Visualization Toolkit (VTK)](https://vtk.org) format to store geometry information to file, and [Paraview](https://www.paraview.org) for visualisation. This allows the geometrical information to be viewed but does not easily permit material information from the model to be checked. A tighter integration is required.

**Skills required:** Python, familiarity with VTK and Paraview would beneficial.

**Difficulty:** Medium

**Mentor(s):** Dr Craig Warren (craig@gprmax.com)


## 5. Web-based framework for GPR data processing

The aim of this project is to develop a web-based framework that allows data output from the simulations to be visualised and processed within a web browser.

Output data from electromagnetic simulations often requires to be processed using the same algorithms and workflows as data that has been acquired from sensors in the lab or field. The data can be a time-varying waveform or 2D/3D image. A variety of platform-specific processing tools current exist that use custom file formats. This does not permit easy sharing and comparison of data. A web-based framework for processing would be a platform-agnostic tool and could be designed to permit users to design and use their own processing plugins around a basic core. A local/remote processing model would lightweight processing tasks to be handled locally and more intensive tasks to be offloaded to remote/cloud compute.

**Skills required:** Python, web frameworks for 2D/3D visualisation.

**Difficulty:** Medium

**Mentor(s):** Dr Craig Warren (craig@gprmax.com) and Mr Dimitris Angelis (dimitris.angelis@northumbria.ac.uk)


## 6. Web-based framework for model building

The aim of this project is to develop a web-based framework that allows models to be graphically built.

Many models, especially for Ground Penetrating Radar (GPR), can be easily specified using a text-based input file, which is currently what is done. This approach can be beneficial when executing large simulations in [high-performance computing (HPC)](https://en.wikipedia.org/wiki/Supercomputer) environments. However, there are also simulations that require fine, complex details to be modelled or where existing geometries already exist (for example in CAD format). In these cases a graphical-based model building environment would be beneficial, and one that does not require bespoke software to be installed. A web-based framework would therefore be a very useful for model building and construction.

**Skills required:** Python, web frameworks for 2D/3D visualisation.

**Difficulty:** Medium

**Mentor(s):** Dr Antonis Giannopoulos (antonis@gprmax.com) and Dr Craig Warren (craig@gprmax.com)


## 7. Modelling complex materials

The aim of this project is to couple and enhance a series of scripts that have been developed to allow materials with complex (frequency dependent) properties to be modelled.

Often materials that required to be simulated have complex electromagnetic properties that can be frequency dependent. There are several models that can be used to simulate different behaviours, and we have developed an initial [series of scripts](https://github.com/gprMax/gprMax/pull/125) that capture this. However work is required to enhance them (possibly with the option for some graphical input) and couple them with gprMax.

**Skills required:** Python, NumPy, some knowledge of electromagnetic wave propagation.

**Difficulty:** Medium

**Mentor(s):** Dr Iraklis Giannakis () and Dr Antonis Giannopoulos (antonis@gprmax.com)


## 8. Importing geometrical information from laser scanners

The aim of this project is to import geometric data acquired from terrestrial laser scanners. The ability to directly model real objects and topographies without entering their geometries manually would be very useful development.

A laser scanner takes distance measurements in every direction to rapidly capture the surface shape of objects, buildings and landscapes. This information is then used to construct a full 3D model of the object. The data from the 3D model requires to be mapped/converted/translated onto the [Finite-Difference Time-Domain (FDTD)](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) grid upon which gprMax is based.

**Skills required:** Python

**Difficulty:** Medium

**Mentor(s):** Dr Craig Warren (craig@gprmax.com) and Dr Iraklis Giannakis ()
