import pandas as pd
import numpy as np
import pickle
from typing import List
import GPy

def time_warping(
    t
):
    """log-Time warping function

    Args:
        t (np.array): timepoints vector
    
    Returns: 
        (np.array): warped timepoints
    """
    return np.log(t)

def log_plus_one(
    x
):
    """Log transformation used as preprocessing
    for the gene expression values

    Args:
        t (np.array): gene expressions vector

    Returns:
        (np.array): log preprocessed gene expressions
    """
    return np.log(x+1)

def partition(collection):
    """Produce all the possible partitions of a set of items
    
    Args:
        collection (List): list of items
    
    Returns: 
        (List): all the possible partitions of collection
    """
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsetsN
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def get_label(subset):
    """Extracting the label of the conditions 
    present in a subset of a partition
    
    Args:
        subset: subset 
    
    Returns: 
        string of comma concatenated conditions
    """
    label = ""
    i = 0
    for condition in subset[:-1]:
        if (i!=0) and ((i%2)==0):
            label+="\n"
        label += condition + ", "
        i+=1
    label += subset[-1]
    
    return label


def save_gp_model(
    gp: GPy.models.GPRegression,
    file_name: str,
):
    """Utility to save a GPy model as a pikle file

    Args:
        gp (GPy.models.GPRegression): the GPy model to be saved
        file_name (str): a file path 
    """
    file = open(file_name, "wb")
    pickle.dump(gp, file)
    #gp.save_model(name)


def load_gp_model(
    file_name:str
):
    """Utility to load a GPy model saved as pickle

    Args:
        file_name (str): model file path

    Returns:
        GPy.models.GPRegression: the pickled GPy model
    """
    file = open(file_name, "rb")
    gp = pickle.load(file)
    return gp


def get_partition_mat(
    partition_num: np.array,
    get_specular: bool=False,
):
    """Assuming that the N samples are organized
    in the data as in partition_num, then this function returns
    two lists of matrices. The lists contain one matrix for each
    partition number (numpy.unique(partition_num)):
        1) a matrix with ones where the partition numbers match 
        2) a matrix with ones where the partition numbers do not match,
          but only looking at the the rows where they match

    e.g., partitio_num = [1,2,3,1,2,3]
    we assume to put this list matching with the rows and columns 
    of a matrix:
          [1,2,3,1,2,3]
       [1,
        2,
        3,
        1,
        2,
        3]
    then the first elements for the two lists are the following
    1) where they match:
        [[1 0 0 1 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]
         [1 0 0 1 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
    2) where they don't, but looking only at the rows where they match:
        [[0 1 1 0 1 1]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]
         [0 1 1 0 1 1]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
    
    This is a utility for the centered kernel calculations.
          

    Args:
        partition_num (np.array): array containing the partition number
            for each sample in the gene expression dataset
        get_specular (bool, optional): to get both the the matrices lists (True)
        or only the list number 1), as explained above (False) . Defaults to False.

    Returns:
        (np.array, np.array): the resulting matrices lists
    """
    subset_idx = np.unique(partition_num)
    subset_mat_list = []
    specular_mat_list = []
    for idx in subset_idx:
        subset_idx_arr = partition_num * (partition_num == idx)
        subset_mat = np.repeat(
            subset_idx_arr.reshape(-1, 1),
            len(subset_idx_arr),
            axis=1
        )
        subset_bool_mat = subset_mat == subset_mat.T

        specular = np.logical_xor(
            subset_mat,
            subset_bool_mat,
        )*(1-subset_bool_mat)
        specular_mat_list.append(specular)

        res = subset_bool_mat * subset_mat
        subset_mat_list.append(res / idx)

    if get_specular:
        return subset_mat_list, specular_mat_list
    else:
        return subset_mat_list