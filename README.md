# isosurface-analysis
This repository contains the code to compute Isosurface Similarity Maps (ISM), representative Isovalues, Distance Equalization, and time-varying multi-variate 
similarity maps (TSM & VSM).

# Requirements
* GCC/G++ Version 6+
* GCC Version 5 for CUDA part
* CUDA
* Thorserialize (https://github.com/Loki-Astari/ThorsSerializer)
* SDL2

#### Tested with
* GCC/G++ 6.2.0
* GCC 5.2.0
* Cuda 8.0


# Program Modules
The built executable contains several modules. To run, use ./isosurfaces <module> <module-input>

The following modules are available:

multi		 - 	 computes the VSM & TSM
single		 - 	 computes a single ISM
distance	 -	 computes a single distance equalization
script		 - 	 computes a set up ISM or Distances
stitch		 - 	 stitches together several json files

There are further modules in the program, but they are for testing purposes only and left undocumented.


## Module Input

Different modules need different input files.

Single & Distance need a file specifying a single volumetric data point (a certain time step & variable) (see configs/single.json)

Script needs a json file as input, that describes a volumetric data set (see configs/ionization-distance-all.json)

Multi uses a short input file specifying where to read the output from an ensemble of "Single" runs (see configs/multi.json)

Stitch uses a list of json files as input. The first one is the so-called "host file" and should contain most entries. The entries of the other json files will be added to this one, and the result will be saved in a separate json file. 
