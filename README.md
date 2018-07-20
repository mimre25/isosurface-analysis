# isosurface-analysis
This repository contains the code to compute Isosurface Similarity Maps (ISM), representative Isovalues, Distance Equalization, and time-varying multi-variate 
similarity maps (TSM & VSM).

For details please refer to our papers [Efficient GPU-accelerated computation of isosurface similarity maps](https://www.computer.org/csdl/proceedings/pacificvis/2017/5738/00/08031592-abs.html) and [Identifying nearly equally spaced isosurfaces for volumetric data sets](https://www.sciencedirect.com/science/article/pii/S0097849318300220):
```bibtex
@inproceedings{imre2017efficient,
  title={Efficient GPU-accelerated computation of isosurface similarity maps},
  author={Imre, Martin and Tao, Jun and Wang, Chaoli and others},
  booktitle={2017 IEEE Pacific Visualization Symposium (PacificVis)},
  pages={180--184},
  year={2017},
  organization={IEEE}
}
```
```bibtex
@article{imre2018identifying,
  title={Identifying nearly equally spaced isosurfaces for volumetric data sets},
  author={Imre, Martin and Tao, Jun and Wang, Chaoli},
  journal={Computers \& Graphics},
  volume={72},
  pages={82--97},
  year={2018},
  publisher={Elsevier}
}
```

# Requirements
* GCC/G++ Version 6+
* GCC Version 5 for CUDA part
* CUDA
* ThorSerializer (https://github.com/Loki-Astari/ThorsSerializer)
* SDL2

#### Tested with
* GCC/G++ 6.2.0
* GCC 5.2.0
* Cuda 8.0


# Program Modules
The built executable contains several modules. To run, use ./isosurfaces <module> <module-input>

The following modules are available:

* multi		 - 	 computes the VSM & TSM
* single		 - 	 computes a single ISM
* distance	 -	 computes a single distance equalization
* script		 - 	 computes a set up ISM or Distances
* stitch		 - 	 stitches together several json files

There are further modules in the program, but they are for testing purposes only and left undocumented.

## Module Input

Different modules need different input files.

Single & Distance need a file specifying a single volumetric data point (a certain time step & variable) (see configs/single.json)

Script needs a json file as input, that describes a volumetric data set (see configs/ionization-distance-all.json)

Multi uses a short input file specifying where to read the output from an ensemble of "Single" runs (see configs/multi.json)

Stitch uses a list of json files as input. The first one is the so-called "host file" and should contain most entries. The entries of the other json files will be added to this one, and the result will be saved in a separate json file. 
