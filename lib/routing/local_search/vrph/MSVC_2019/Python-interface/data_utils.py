# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance_matrix
import pickle
import os, sys
from pathlib import Path

import VRPH
from VRPH import VRP


def save_python_object(object_to_save, filename, dirname=''):
    """pickles a python object. Uses .pkl as file extension."""
    if len(filename) > 3:
        if not filename[-4:] == '.pkl':
            filename = filename + '.pkl'
    else:
        filename = filename + '.pkl'
    path = os.path.join(dirname, filename)
    
    #create directory if neccessary
    Path(dirname).mkdir(parents=True, exist_ok=True)

    file_to_store = open(path, "wb")
    pickle.dump(object_to_save, file_to_store)
    file_to_store.close()

def load_python_object(filename, dirname=''):
    """Unpickles an object from file."""
    path = os.path.join(dirname, filename)
    file_to_read = open(path, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    return loaded_object

def generate_tsp(graph_size, num_instances=1, seed=1234, save_to_file=False, name='', dirname=''):
    """
    generates tsp graphs

    Parameters
    ----------
    graph_size : int
    num_instances : int, optional. The default is 1.
    seed : int, optional. The default is 1234.

    Returns
    -------
    instances : list of dictionaries containing coordinates, the depot index and number of vehicles
    """
    
    np.random.seed(seed=seed)
    assert graph_size > 1, 'graph_size must be >= 2'
    
    instances = []
    for i in range(num_instances):
        data = {}
        data['num_vehicles'] = 1
        data['depot'] = 0
        data['coordinates'] = np.random.uniform(low=0.0, high=1.0, size=(graph_size, 2))
        instances.append(data)
    
    if save_to_file:
        if not name == '':
            name = '_' + name
        filename = 'TSP{}_num{}_seed{}{}'.format(graph_size, num_instances, seed, name)
        save_python_object(instances, filename, dirname)
    
    return instances

def generate_cvrp(graph_size, num_vehicles = 1, num_instances=1, seed=1234, save_to_file=False, name='', dirname=''):
    """
    generates cvrp graphs

    Parameters
    ----------
    graph_size : int
    num_depots : int, optional. The default is 1.
    num_instances : int, optional. The default is 1.
    seed : int, optional. The default is 1234.

    Returns
    -------
    instances : list of dictionaries containing coordinates, the depot index, the demands, and number of vehicles 
        DESCRIPTION.
    """
    
    np.random.seed(seed=seed)
    assert graph_size > 1, 'graph_size must be >= 2'
    
    instances = []

    if graph_size<=10:
        vehicle_capacity = 20.
    elif graph_size<=20:
        vehicle_capacity=30.
    elif graph_size<=50:
        vehicle_capacity=40.
    else:
        vehicle_capacity=50.

    for i in range(num_instances):
        data = {}
        data['depot'] = 0
        data['coordinates'] = np.random.uniform(low=0.0, high=1.0, size = (graph_size +1, 2))
        data['vehicle_capacities'] = np.full(shape = num_vehicles, fill_value = vehicle_capacity, dtype=np.int)
        data['num_vehicles'] = num_vehicles
        data['demands'] = np.random.uniform(1, 10, size = graph_size)
        data['demands'] = np.insert( data['demands'], 0, 0)
        instances.append(data)
    
    if save_to_file:
        if not name == '':
            name = '_' + name
        filename = 'CVRP{}_num{}_seed{}{}'.format(graph_size, num_instances, seed, name)
        save_python_object(instances, filename, dirname)

    return instances

def calc_distance_matrices(instances, p=2):
    """
    This policy calculates the distance matrices for all instances given with scipy.

    Parameters
    ----------
    instances : list
        list of instances; An Instance is a dict that has to have the 'coordinates' key.
    p : int, optional
        the p value of the Minkowski metric for scipy. Default is 2 which corresponds to the euclidian distance.

    Returns
    -------
    instances : list
        returns the list of instances with the calculated distance matrices.

    """
    for i in instances:
         i['distance_matrix'] = distance_matrix(i['coordinates'], i['coordinates'], p=p)
    return instances

def scale_distance_matrices(instances, scale_factor=10**5):
    """
    This function scales the distances matrices of instances by scale_factor and casts them as an integer which is better to use with OR Tools

    Parameters
    ----------
    instances : list
        list of instances; An Instance is a dict that has to have the 'distance_matrix' key.
    scale_factor : number, optional
        The factor to scale. The default is 10**6. Should be sufficiently large to not add too much error when casting to int.

    Returns
    -------
    instances : list
        returns the list of instances with the scaled and casted distance matrices.

    """
    for i in instances:
         i['distance_matrix'] = (scale_factor*i['distance_matrix']).astype(int)
    return instances

class TSP_Generator():
    def __init__(self, graph_size: int, seed: int = 1234):
        """An object from which TSP Instances can be sampled.

        Args:
            graph_size (int): Number of points for the TSP Graph
            seed (int, optional): Numpy random Seed. Defaults to 1234.
        """
        assert graph_size >= 2, 'graph_size must be >= 2'
        self.graph_size = graph_size
        self.seed = seed
        np.random.seed(seed=seed)

    def sample_instance(self, ):
        data = {}
        data['customers'] = np.random.uniform(low=0.0, high=1.0, size=(self.graph_size, 2))
        data['type'] = "TSP"
        return data


def load_vrph (model, data):
    
    if data["type"] == "TSP":
        # the load problem functions with
        # (type, coordinates, demands, [best_distance, num_nodes, capacity, max_route_length, neighbor], edge_type, edge_format)
        # set -1 for any missing value
        model.load_problem( 0, data['customers'].tolist(), [float(-1)], [float(-1), float(data["customers"].shape[0]), float(-1), float(-1), float(0)], 9, -1)
    
def augment_data(data, N = 1 ,augment_type = "both"):
    """Produce augmented variations of the same VRP instance using three techniques: rotate, shift, both.

    Args:
        data (array): a tensor containing the data to augment
        N (int): number of desired maximum outputed augmentations
        augment_type (string): augmentation policy
    """




    return data
    
    
    
    
