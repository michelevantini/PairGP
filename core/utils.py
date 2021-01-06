import pandas as pd
import numpy as np
import pickle

def time_warping(t):
    '''
    Time warping function
    :param t: timepoints vector
    :return: warped timepoints
    '''

    return np.log(t)

def log_plus_one(x):
    return np.log(x+1)

def partition(collection):
    '''
    Produce all the possible partitions of a set of items
    :param collection: an iterable item 
    :return: all the possible partitions of collection
    '''
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
    '''
    Extracting the label of the conditions 
    present in a subset of a partition
    :param subset: subset 
    :return: string of comma concatenated conditions
    '''
    label = ""
    i = 0
    for condition in subset[:-1]:
        if (i!=0) and ((i%2)==0):
            label+="\n"
        label += condition + ", "
        i+=1
    label += subset[-1]
    
    return label


def save_gp_model(gp, file_name):
    file = open(file_name, "wb")
    pickle.dump(gp, file)
    #gp.save_model(name)


def load_gp_model(file_name):
    file = open(file_name, "rb")
    gp = pickle.load(file)
    return gp


def get_partition_mat(partition_num, get_specular=False):
    subset_idx = np.unique(partition_num)
    subset_mat_list = []
    specular_mat_list = []
    for idx in subset_idx:
        subset_idx_arr = partition_num * (partition_num == idx)
        subset_mat = np.repeat(subset_idx_arr.reshape(-1, 1), len(subset_idx_arr), axis=1)
        subset_bool_mat = subset_mat == subset_mat.T

        specular = np.logical_xor(subset_mat, subset_bool_mat)*(1-subset_bool_mat)
        specular_mat_list.append(specular)

        res = subset_bool_mat * subset_mat
        subset_mat_list.append(res / idx)

    if get_specular:
        return subset_mat_list, specular_mat_list
    else:
        return subset_mat_list