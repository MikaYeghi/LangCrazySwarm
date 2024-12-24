# LangCrazySwarm

## Installation

First, install the Conda environment in the following way:
```
conda env create -f environment.yml -n langcrazyswarm
conda activate langcrazyswarm
```

Once you are done with the environment installation, following the [Crazyswarm installation](https://crazyswarm.readthedocs.io/en/latest/installation.html) for Physical Robots and Simulation with the environment activated. After that, install PyKDL and PyQT5 in the following way:
```
conda install conda-forge::python-orocos-kdl
pip install PyQT5
```

<!-- ## VisPy Visualization (optional)

If you wish to be able to visualize in VisPy, add the following lines of code to `visVispy.py` in `crazyswarm/scripts/pycrazyswarm/visualizer`: -->

## Producing animations of paths

To produce animations of the paths, run the following command:
```
python process_data.py
```

Make sure you delete the first 3 rows in the Vicon data CSV file, otherwise `pandas` will not be able to read the file properly.