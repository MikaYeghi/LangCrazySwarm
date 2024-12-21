# LangCrazySwarm

### Installation

Follow the Crazyswarm installation. After that, install PyKDL and PyQT5 in the following way:
```
conda install conda-forge::python-orocos-kdl
pip install PyQT5
```

After that, add the following lines of code to `visVispy.py` in `crazyswarm/scripts/pycrazyswarm/visualizer`. This action will allow visualization in vispy.

### Producing animations of paths

To produce animations of the paths, run the following command:
```
python process_data.py
```

Make sure you delete the first 3 rows in the Vicon data CSV file, otherwise `pandas` will not be able to read the file properly.