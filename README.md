# Cognitive Route Planning for Cycling

We have augmented a cognitive route algorithm by Dubey et al. (2023) [[1]](#1) for the context of cycling and tested iy for the city of Zurich. The basic workflow is illustrated in the following figure.
![cognitive routing.png](cognitive%20routing.png)

It is worth noting that the code is not yet fully optimized for general use and only works with a particularly simplified road network based on Ballo et al. (2024) [[2]](#2) and enrichment data from the city of Zurich. However, the simplified network is derived from Open Street Maps. Additionally, we enrich the network with OD trip counts and public space significance data. Therefore, in order to run the code with your own network, multiple parts should be adjusted. 

### Setup Instructions

Ensure you have Python 3.11 installed. If not, download it from Python's official website. Clone this repository and navigate into the project directory.
Run `pip install -r requirements.txt` to install required external dependencies. 

### File description

* `paths_example.ini`: example config file with: 1) path to network data, nodes and edges, 2) path to enrichment data 3) output directory to store plots 4) path to credentials to log in to database that contains trajectory data. The functions use geopandas GeoDataFrames, hence trajectories can be directly provided in such format.
* `network_processing.py`: Functions to preprocess the network data and enrich it with predefined network metric and saliency values.
* `trajectory_processing.py`: Functions to retrieve and process the trajectory data from the database.
* `clustering.py`: Functions to generate hierarchical clusters mirroring three common spatial knowledge representations.
* `memory_distortions.py`: Functions to distort the spatial memory and the locations of network and abstract nodes based on the cluster centroids and the saliency values.
* `cognitive_routing.py`: Functions to calculate the cognitive route based on the distorted spatial memory and the enriched network data.
* `plotting.py`: Functions to visualize the results of the cognitive route planning.
* `similarity_comparison.py`: Functions to compare the similarity of the cognitive route to the actual route.
* `run_dynamic_cognitive_routing.py`: runs the dynamic cognitive route planning for a specific trajectory.

## References
<a id="1">[1]</a> 
Dubey, R. K., Sohn, S. S., Thrash, T., Hölscher, C., Borrmann, A., & Kapadia, M. (2023). Cognitive Path Planning With Spatial Memory Distortion. IEEE Transactions on Visualization and Computer Graphics, 29(8), 3535–3549. https://doi.org/10.1109/TVCG.2022.3163794

<a id="2">[2]</a>
Ballo, L., & Axhausen, K. W. (2024). Modeling sustainable mobility futures using an automated process of road space reallocation in urban street networks: A case study in Zurich. In 103rd Annual Meeting of the Transportation Research Board (TRB 2024).