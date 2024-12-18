# CT_TSP_On-demand_services

This repository contains details on my attempt to try to learn, understand and apply the TSP to the city of calgary. I hope to use this to help with on-demand services for calgary transit. 

Problem Statement: Want to create an analytical model to equate different types of networks within the city. Want to identify different K values and spacial demand distribution around transit hubs (potentially). Ring radial as a distance metric (confirm what this means).

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

## Current Progress
Utilized geocoders library and created a map of calgary utilizing manually written data.
Transit specific functions have failed...

### Feedbackk
- Trim the scope: pick a few transit hubs around the city (NP, Tuscany, crowfoot, anything with park and ride) between 3 and 5 hubs. 
  - Plot on that graph. Test a dozen differnt cases (30 instances within a different number)
  - Graph on X axis N and distance to complete journey on Y axis for a radius (cap 4 to 16)
    - Want a sublinear plot that should be related to the root of the equation that allows us to estimate the K value
    - K value should be different for different stations
    - Hypothesize the different K values based on different parameters (road network, )
    - Gives good estimation value to the city
    - Instead of doing a coordinate system, use a polar system. Random degree from the terminal. You can constrain the random part of the radial portion. 
    - Currently closer to current TSP setup
    - Could mention a theoretical donut implementation.
- ARC MAP way to deal with random mapping problems. Confirm the name of the map function
- weight it to be less uniformally random
- Conduct and justify your work using a literature review
- Doing analysis first
- 200
https://www.sciencedirect.com/science/article/pii/S0191261508000969
https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=vPW-ZhkAAAAJ&cstart=200&pagesize=100&sortby=pubdate&citation_for_view=vPW-ZhkAAAAJ:d1gkVwhDpl0C

Donut Implementation:
https://jordinl.com/posts/2019-02-15-how-to-generate-random-geocoordinates-within-given-radius

## Park and Ride Termninals I am going to Focus on
- Priority: 
  - Outskirt of city
  - Parking > 850
  - Accessibility > X
  - Train Access Yes/No?



Tuscany Station - Rocky Ridge Terminal [NW deep]
Douglas Glen Terminal [deep SE]
BR Barlow Stn Bus Terminal [SE]
North Pointe Terminal A
McKnight-Westwinds CTrain Park and Ride
Canyon Meadows Park & Ride
Sirocco Park and Ride

SOUTH 
Fish Creek - Lacombe CTrain Station [lat, long] = [50.9233027, -114.0758714]
Somerset-Bridlewood [50.8986570, -114.0699560]

NORTH:
Crowfoot - North and South [51.1226635, -114.2070422]
Tuscany Station - Rocky Ridge Terminal [51.1350249, -114.2348192]


EAST - WEST
Saddletowne CTrain Park and Ride [51.1254114, -113.9487872]
Sirocco Station [51.0381335, -114.1695420]
marlborough mall [51.0552984, -113.9786681]

BUSSES:
North Pointe [51.1570621, -114.0680757]
