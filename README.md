# KShape-cpp
Implementation of KShape clustering using Eigen 3 in C++

The code is currently in functional draft form, requires more practical packaging

# Requirements
KShape-cpp was tested with the following software

1. Ubuntu 20.04.2 LTS
2. GCC 9.3.0
3. Eigen 3.3.9 

# Installation and usage

Download Eigen 3 from http://eigen.tuxfamily.org/index.php?title=Main_Page#Download (this is tested aand working with Eigen 3.3.9)

Unzip and extract the Eigen directory to /usr/local/include/, and the unsupported directory to /usr/local/include/Eigen

Compilation can be done simply as 
```console
g++ kshape.cpp -o kshape.o -O2
```
and execution as 
```console
./kshape.o FILE.csv  [ROWS] [COLS] [k] [RUNS]
```
Where ```k``` clusters are formed from data specified in ``FILE.csv``, containing ```ROWS``` series each of length ```COLS```. 

```RUNS``` iterations of the algorithm will be run, with the lowest overall distance result being kept

# Output
Two csv files will be outputted, ```out_centroids.csv``` a ```k```x```COLS``` csv containing z-normalised centres, and ```out_indices.csv``` a 1-row csv containing the index of the cluster that each input series is allocated to. 
