# BASE IMPORT
import os
import pickle
import shutil
from abc import abstractmethod
from typing import List

# EXTERNAL IMPORT
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

# MODEL IMPORT
import GPy
from utils import (
    get_label,
    get_partition_mat,
    log_plus_one,
    load_gp_model,
    save_gp_model,
    partition,
    time_warping
) 

from PairingEffectKernel import PairingEffectKernel
#from BaseKernel import BaseKernel

# Remove comments here if you don't
# want to see any wornings
# import warnings
# warnings.filterwarnings("ignore")

# Add some personal settings here for 
# visualization purposes
plt.style.use('seaborn-poster')
matplotlib.rc('axes', titlesize=24)
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)

'''
Definition of the base class which will be the base
for the pairing effect model and the base model
'''
class DiffConditionModel:
    def __init__(
        self,
        gene_data: pd.DataFrame,
        gene_name: str,
        timepoints: np.array,
        conditions: list,
        n_replicates: int,
        models_folder: str="./models",
        hyperparams_iter: int=5,
        colors: List[str]=None,
        subcolors: List[List[str]]=None,
        repcolors: List[str]=None,
        markers: List[str]=None,
    ):
        """
        Args:
            gene_data (pd.DataFrame): gene data in a pandas Series format
                The data format accepted in input is a pandas Series (or one row from a DataFrame).  
                If coming from a pandas DataFrame, the name of the columns must be organized with 
                the following structure:
                    condition_timepoint_replicatenum
                So the structure is the name of the condition, the time point (without any sort of
                indication such as "h" is the time is in hours), and then a number the identifies 
                the replicate from which the data point comes from.
                The data points need to be already expressed in the unit that one wants to use. 
                In other words, any sort of transformation on the gene read-counts need to be 
                applied to the dataset before passing the datapoints to this software.
            gene_name (str): the name of the gene that will be diplayed in the plots
            timepoints (np.array): list of timepoints used in the gene expression time-series
                These need to match with the timepoints in the columns name notation 
                for gene_data
            conditions (list): list of the conditions (or tretments) in the data.
                The conditions need to be listed with the same name used in the columns name 
                notation for gene_data
            n_replicates (int): total number of replicates used in gene_data.
                These need to match with the number of replicates reported in the columns
                name notation for gene_data
            models_folder (str): path to the folder where the model files will be saved
            hyperparams_iter (int): number of times the hyperparameter optimization
                will be run during the model fit
            colors (List[str]): list of colors in a format accepted by matplotlib.pyplot
                one for each of the conditions in the dataset, required for the plotting
                e.g., with 6 conditions
                    ["#00334e", "#801336", "#12d3cf", "#f18c8e", "k", "r"] 
                You can find an example of colors for a dataset with 5 conditions and 3 replicates
                in the Jupyter notebook "notebooks/notebook.ipynb"
            subcolors (List[List[str]]): list of list of colors in a format accepted by matplotlib.pyplot
                the list must contain one list for each condition
                the inner list must contain one color for each replicate
                e.g., 5 conditions with 3 replicates
                    [
                        ["#00334e", "#145374", "#5588a3"],
                        ["#801336", "#c72c41", "#ee4540"],
                        ["#12d3cf", "#67eaca", "#b0f4e6"],
                        ["#f18c8e", "#f0b7a4", "#f1d1b5"],
                        ["#00334e", "#145374", "#5588a3"],
                    ]
            repcolors=None (List[str]): list of colors in a format accepted by matplotlib.pyplot
                the list must contain one color for each replicate
                e.g., with 3 replicates
                    ["#bfcd7e", "#4592af", "#a34a28"
            markers=None (List[str]): list of markers in a format accepted by matplotlib.pyplot
                the list must contain one merker for each replicate in the dataset
                e.g., with 3 replicates
                    ["o", "x", "*"]
        """
        self.conditions = conditions
        # we generate here the list of all the possible partitiions
        # of the condition set
        self.partitions = list(partition(conditions))
        # to speed up the computation we create a dictionary
        # where we assign a unique id to each condition
        self.conditions_dict = self.get_conditions_dict(conditions)
        # we create a more descriptive version of the gene_data
        # where we store replicate number, condition and timepoint
        # in separate columns to speed up the computation
        self.gene_data = self.augment_gene_data(gene_data)
        self.gene_name = gene_name
        self.timepoints = timepoints
        self.hyperparams_iter = hyperparams_iter
        self.n_replicates = n_replicates

        # if the model folder doesn't exist
        # we create it
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.models_folder = models_folder

        # Decide whether to use the already stored
        # models to select the best model or to delete
        # them and (re-)building them
        self.use_stored=False        
        # RBF Kernel can be run on GPU with GPy
        self.use_gpu=False
        
        # we will store here the ranking of the 
        # models for each of the possible partitions
        # of the condition set
        self.partitions_ranking = None

        # colors settings to display 
        #self.colors = ["#00334e", "#801336", "#12d3cf", "#f18c8e", "k", "r"]
        self.colors = colors
        self.subcolors = subcolors
        self.repcolors = repcolors
        self.markers = markers

        # number of points to create between timepoint[0] and 
        # timepoints[-1] to visualize the predictions
        self.TIME_PRED_LEN = 100


    def get_gene_data(self):
        return self.gene_data


    @abstractmethod
    def fit(self):
        pass

       
    def gp_unit(
        self,
        X: np.array,
        y: np.array,
        kernel: GPy.kern.Kern,
        normalizer: bool=False,
    ) -> (GPy.models.GPRegression, float):
        """Given the timepoints X and the gene expression
        data for one gene y, then fit a GP regression model
        with the given kernel.

        Args:
            X (np.array): timepoints for the gene expression time series
            y (np.array): gene expression time series data
            kernel (GPy.kern.Kern): kernel to use in the GP regression model
            normalizer (bool, optional): Normalizer to be passed to GPy.models.GPRegression. Defaults to False.

        Returns:
            (GPy.models.GPRegression, float): the fitted model and the optimal log marginal likelihood
        """
        gp = GPy.models.GPRegression(X, y, kernel, normalizer=normalizer)
        gp.optimize_restarts(num_restarts=self.hyperparams_iter, verbose=False)
        score = gp.log_likelihood()
        return gp, score


    def get_conditions_data(
        self,
        subset: list,
        normalize: bool=False,
    ) -> (pd.DataFrame, float, float) :
        '''Filter data to return only the data related to subset
        This function also applies the proper transformation to the data
        
        Args:
            subset (list): subset of conditions to filter
            normalize (bool): whether to normalize the data or nott
        
        Returns:
            (pd.DataFrame, float, float): normalized dataset, mean and variance 
                used in the normalization
        
        '''
        conditions_idx = [self.conditions_dict[condition] for condition in subset]
        data = self.gene_data[self.gene_data['c'].isin(conditions_idx)]

        if self.timewarping:
            data['X'] = time_warping(data['X'])

        if normalize:
            mu, sigma = np.mean(data['y']), np.std(data['y'])
            data['y'] = (data['y'] - mu) / sigma
        else:
            mu, sigma = None, None

        return data, mu, sigma


    def get_time_pred(self) -> np.array:
        """Calculate the list of timepoints for calculating
        predictions. We use numpy.linspace to get self.TIME_PRED_LEN
        timepoints equally spaced from the first timepoints to the last.

        Returns:
            np.array: list of timepoints for calculating predictions
        """

        # if we are using the timewarping then we need to apply it to 
        # the timepoints for the predictions as well
        if self.timewarping:
            return np.linspace(
               time_warping(self.timepoints[0]),
               time_warping(self.timepoints[-1]),
               self.TIME_PRED_LEN
            )
        else:
            return np.linspace(
                self.timepoints[0],
                self.timepoints[-1],
                self.TIME_PRED_LEN
            )


    @abstractmethod
    def plot(self):
        '''
            Each class inheriting from DifConditionModel
            needs to implement a method to visualize the model fit
        '''
        pass

    
    def plot_data(
        self,
        timewarping: bool=False,
        logexpr: bool=False,
    ):
        """Utility to plot the data contained in the self.gene_data attribute

        Args:
            timewarping (bool, optional): Apply log-timewarping to the timepoint 
                for the sake of plotting. Defaults to False:bool.
            logexpr (bool, optional): Apply log-trainsformation log(y+1) to the 
                gene expression values for the sake of plotting. Defaults to False:bool.
        """
        data, _, _ = self.get_conditions_data(self.conditions)
        partition_num = self.generate_partition_number([[c] for c in self.conditions])

        replicate_idx = pd.unique(self.gene_data['r']).astype(int)

        data['s'] = partition_num
        #data, mu, sigma = self.normalize_gene_data(data)
        #print(mu)
        data = data.sort_values(["s", "r"])

        X = data[["X", "r"]]
        y = data[["y"]]

        #print(data)
        plt.figure(figsize=(22, 8))
        plt.subplot(121)
        for i, subset in enumerate(self.conditions):

            #plt.subplot(len(self.conditions),1, i+1)
            subset_data = data[data['s'] == (i + 1)]  # self.get_conditions_data(subset)
            for r_idx, r in enumerate(replicate_idx):
                r_data = subset_data[subset_data["r"] == r]
                if logexpr:
                    data_points = r_data["y"].values
                else:
                    data_points = np.exp(r_data["y"].values)-1

                if timewarping:
                    time_points = r_data["X"].values
                else:
                    time_points = np.exp(r_data["X"].values)

                if r_idx==0:
                    plt.scatter(
                        time_points,
                        data_points,
                        marker=self.markers[r_idx],
                        c=self.colors[i],
                        label=str(subset)
                    )
                else:
                    plt.scatter(
                        time_points,
                        data_points,
                        marker=self.markers[r_idx],
                        c=self.colors[i]
                    )
            plt.tight_layout()
            plt.xlabel("Time (hours)")
            plt.ylabel("Gene expression (read count)")
            plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            plt.grid()

        if timewarping:
            data["X"] = time_warping(data["X"])

            X = data[["X", "r"]]
            y = data[["y"]]

            plt.subplot(122)
            #plt.figure(figsize=(16, 8))
            for i, subset in enumerate(self.conditions):
                # plt.subplot(len(self.conditions),1, i+1)
                subset_data = data[data['s'] == (i + 1)]  # self.get_conditions_data(subset)
                for r in range(self.n_replicates):
                    r_data = subset_data[subset_data["r"] == (r + 1)]
                    data_points = np.exp(r_data["y"].values) - 1
                    if r == 0:
                        plt.scatter(r_data["X"].values, data_points, marker=self.markers[r], c=self.colors[i],
                                    label=str(subset))
                    else:
                        plt.scatter(r_data["X"].values, data_points, marker=self.markers[r], c=self.colors[i])
                # plt.grid()

            # plt.tight_layout()
            plt.xlabel("Time")
            #plt.ylabel("Gene read count")
            plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            plt.grid()
        plt.tight_layout()
        plt.show()


    def generate_partition_number(
        self,
        partition: List[List[str]],
    ):
        """Utility to generate the the partition number column
        starting from the data and the partition that we are analysing.

        Args:
            partition (List[List[str]]): a partition of the condition set

        Returns:
            np.array: concatenation of the partition numbers for each 
                timepoint in the data
        """
        partition_num = []
        for i, subset in enumerate(list(partition)):
            condition_no = [self.conditions_dict[condition] for condition in subset]
            subset_data = self.gene_data[self.gene_data['c'].isin(condition_no)]
            tmp = np.repeat(i + 1, len(subset_data))
            #tmp = np.repeat(i + 1, len(self.timepoints) * self.n_replicates * len(subset))

            #print(tmp.shape)
            #print(len(self.timepoints), self.n_replicates, len(subset))
            partition_num.append(tmp)
        partition_num = np.concatenate(partition_num)
        return partition_num


    def get_conditions_dict(
        self,
        conditions
    ):
        """Mapping conditions to integers starting from 1

        Args:
            conditions (List["str]): the list of conditions to map

        Returns:
            dict: dictionary containing the conditions mapping
        """
        conditions_dict = {}
        for i, c in enumerate(conditions):
            conditions_dict[c] = i + 1

        return conditions_dict


    def augment_gene_data(
        self,
        gene_data: pd.Series,
    ):
        """Data are transformed in a more efficient structure. From a
        structure where each point is annotated with condition_timepoint_replicatenum
        we get a pd.DataFrame with 4 columns:
            (timepoint, condition, replicatenum, value) 
        the actual column names will be:
            (X, c, r, y)
        This will allow a faster computation during the model selection process.

        Args:
            gene_data (pd.Series): pandas dataframe with the gene expression data
                The data in should be in the format specified in the constructor of this class. 

        Returns:
            pd.DataFrame: tranformed DataFrame
        """
        if isinstance(gene_data, pd.DataFrame):
            cols = gene_data.columns
        elif isinstance(gene_data, pd.Series):
            cols = gene_data.index

        X = []
        r = []
        c = []
        for col_name in cols:
            condition, t, rep = col_name.split("_")
            X.append(float(t))
            r.append(int(rep))
            c.append(self.conditions_dict[condition])

        X = np.array(X).reshape((-1,1))
        r = np.array(r).reshape((-1,1))
        c = np.array(c).reshape((-1, 1))
        y = gene_data.values.reshape((-1,1))
        df = pd.DataFrame(np.concatenate((X, c, r, y), axis=1), columns=["X", "c", "r", "y"]).sort_values(["c", "X"])
        return df


    def delete_models(
        self
    ):
        """Utility to automatically eliminate the model files 
        generated during the model selection process.
        The models are eliminated from self.models_folder
        """
        for the_file in os.listdir(self.models_folder):
            file_path = os.path.join(self.models_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

