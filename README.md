# CT_TSP_On-demand_services

This repository contains details on my attempt to try to learn, understand and apply the TSP to the city of calgary. I hope to use this to help with on-demand services for calgary transit. 

The tools available so far: 
- `python-tsp` library that can be found at: https://pypi.org/project/python_tsp/ and https://github.com/fillipe-gsm/python-tsp
  - `pip install python-tsp`
  - `poetry add python-tsp`  # if using Poetry in the project
  - This would be an individual use case where i will need to explore and understand how to use TSPLIB files `.tsp`
  - Can i use this to find a model to map these services to calgary transit
  - Best way would be to find bus hubs and the types of structures around it euclidean, rectilinear etc
  - Distance Matrix Creation: `numpy` or `geopandas`
    - `geopandas` relies on `shapely` and `pyogrio`
- Data Processing and Analysis: `numpy` `pandas` `SciPy` `geopandas`
- Optimization and Simulation modelling: `PuLP` `Google OR-Tools` `SimPy`
- Spacial Data and Visualization: `QGIS`, `MATPLOTLIB`, `Seaborn`
- Heuristic Optimization: `Pytorch`, `tensorflow` or ` DEAP library`
