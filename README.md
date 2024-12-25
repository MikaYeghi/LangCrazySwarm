# LangCrazySwarm

## Installation

#### Environment

First, install the Conda environment in the following way:
```
conda env create -f environment.yml -n langcrazyswarm
conda activate langcrazyswarm
```

#### CrazySwarm

Once you are done with the environment installation, following the [Crazyswarm installation](https://crazyswarm.readthedocs.io/en/latest/installation.html) for Physical Robots and Simulation with the environment activated. After that, install PyKDL and PyQT5 in the following way:
```
conda install conda-forge::python-orocos-kdl
pip install PyQT5
```

#### LangCrazySwarm

After finishing the CrazySwarm installation, copy the `run_swarm.py` script into CrazySwarm's scripts. From CrazySwarm's root directory, this folder is located at `ros_ws/src/crazyswarm/scripts/`. Then, add a `.env` file in the same directory which would contain all the API keys required for running workflows with LangGraph. See an example content of `.env` below:

```
# OpenAI API key
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# LangSmith for tracing the workflows
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
LANGCHAIN_PROJECT=YOUR_LANGCHAIN_PROJECT_NAME
```

The bottom 4 rows can be easily obtained from [LangSmith](https://smith.langchain.com/) when initializing a new project.

<!-- ## VisPy Visualization (optional)

If you wish to be able to visualize in VisPy, add the following lines of code to `visVispy.py` in `crazyswarm/scripts/pycrazyswarm/visualizer`: -->

## Producing animations of paths

To produce animations of the paths captured by Vicon, run the following command:
```
python process_data.py
```

Make sure you delete the first 3 rows in the Vicon data CSV file, otherwise `pandas` will not be able to read the file properly.