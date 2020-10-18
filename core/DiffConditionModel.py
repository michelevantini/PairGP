# BASE IMPORT
import os
import pickle
import shutil
from abc import abstractmethod

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
from BaseKernel import BaseKernel

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
    def __init__
        self,
        gene_data: pd.DataFrame,
        gene_name: str,
        timepoints: np.array,
        conditions: list,
        n_replicates: int,
        models_folder="./models": str,
        hyperparams_iter=5: int,
    ):
        """
        Arguments:
            gene_data {pd.DataFrame} -- gene data in a pandas DataFrame format
            gene_name {str} -- the name of the gene that will be diplayed in the plots
            timepoints {np.array} -- list of timepoints used in the gene expression time-series
                These need to match with the timepoints in the columns name notation 
                for gene_data
            conditions {list} -- list of the conditions (or tretments) in the data.
                The conditions need to be listed with the same name used in the columns name 
                notation for gene_data
            n_replicates {int} -- total number of replicates used in gene_data.
                These need to match with the number of replicates reported in the columns
                name notation for gene_data
            models_folder {str} -- path to the folder where the model files will be saved
            hyperparams_iter {int} -- number of times the hyperparameter optimization
                will be run during the model fit
        """
        self.__conditions = conditions
        # we generate here the list of all the possible partitiions
        # of the condition set
        self.__partitions = list(partition(conditions))
        # to speed up the computation we create a dictionary
        # where we assign a unique id to each condition
        self.__conditions_dict = self.__get_conditions_dict(conditions)
        # we create a more descriptive version of the gene_data
        # where we store replicate number, condition and timepoint
        # in separate columns to speed up the computation
        self.__gene_data = self.__augment_gene_data(gene_data)
        self.__gene_name = gene_name
        self.__timepoints = timepoints
        self.__hyperparams_iter = hyperparams_iter
        self.__n_replicates = n_replicates

        # if the model folder doesn't exist
        # we create it
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.__models_folder = models_folder

        # Decide whether to use the already stored
        # models to select the best model or to delete
        # them and (re-)building them
        self.__use_stored=False        
        # RBF Kernel can be run on GPU with GPy
        self.__use_gpu=False
        
        # we will store here the ranking of the 
        # models for each of the possible partitions
        # of the condition set
        self.partitions_ranking = None

        # colors settings to display 
        #self.colors = ["#00334e", "#801336", "#12d3cf", "#f18c8e", "k", "r"]
        self.__colors = ["#00334e", "#801336", "#12d3cf", "#f18c8e", "k", "r"]
        self.__subcolors = [
            ["#00334e", "#145374", "#5588a3"]
            , ["#801336", "#c72c41", "#ee4540"]
            , ["#12d3cf", "#67eaca", "#b0f4e6"]
            , ["#f18c8e", "#f0b7a4", "#f1d1b5"]
            , ["#00334e", "#145374", "#5588a3"]
        ]
        self.__repcolors = ["#bfcd7e", "#4592af", "#a34a28", "#e3c4a8", "#6c5ce7", "#474141"]
        self.__markers = ["o", "x", "*", "d", "+", "v", "^", "s", "<", "."]

        # number of points to create between timepoint[0] and 
        # timepoints[-1] to visualize the predictions
        self.__TIME_PRED_LEN = 100


    def get_gene_data(self):
        return self.__gene_data


    @abstractmethod
    def fit(self):
        pass

       
    def __gp_unit(
        self,
        X: np.array,
        y: np.array,
        kernel: GPy.kern.Kern,
        normalizer=False: bool,
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
        gp.optimize_restarts(num_restarts=self.__hyperparams_iter, verbose=False)
        score = gp.log_likelihood()
        return gp, score


    def get_conditions_data(
        self,
        subset: list,
        normalize=False : bool,
    ) -> (pd.DataFrame, float, float) :
        '''Filter data to return only the data related to subset
        This function also applies the proper transformation to the data
        
        Args:
            subset {list} -- subset of conditions to filter
            normalize {bool} -- whether to normalize the data or nott
        
        Returns:
            (pd.DataFrame, float, float): normalized dataset, mean and variance 
                used in the normalization
        
        '''
        conditions_idx = [self.__conditions_dict[condition] for condition in subset]
        data = self.__gene_data[self.__gene_data['c'].isin(conditions_idx)]

        if self.__timewarping:
            data['X'] = time_warping(data['X'])

        if normalize:
            mu, sigma = np.mean(data['y']), np.std(data['y'])
            data['y'] = (data['y'] - mu) / sigma
        else:
            mu, sigma = None, None

        return data, mu, sigma


    def __get_time_pred(self) -> np.array:
        """Calculate the list of timepoints for calculating
        predictions. We use numpy.linspace to get self.__TIME_PRED_LEN
        timepoints equally spaced from the first timepoints to the last.

        Returns:
            np.array: list of timepoints for calculating predictions
        """

        # if we are using the timewarping then we need to apply it to 
        # the timepoints for the predictions as well
        if self.__timewarping:
            return np.linspace(
               time_warping(self.__timepoints[0]),
               time_warping(self.__timepoints[-1]),
               self.__TIME_PRED_LEN
            )
        else:
            return np.linspace(
                self.__timepoints[0],
                self.__timepoints[-1],
                self.__TIME_PRED_LEN
            )


    def __generate_Xnew(self, time_pred, partition):
        Xnew = np.tile(time_pred, len(partition) * self.__n_replicates)

        Xnew_partition_num = []
        Xnew_replicate_num = []
        for i, subset in enumerate(list(partition)):
            Xnew_partition_num.append(np.repeat(i + 1, len(time_pred) * self.__n_replicates))

            for r in range(self.__n_replicates):
                Xnew_replicate_num.append(np.repeat(r + 1, len(time_pred)))
        Xnew_partition_num = np.concatenate(Xnew_partition_num)
        Xnew_replicate_num = np.concatenate(Xnew_replicate_num)

        Xnew = Xnew.reshape((-1, 1))
        Xnew_partition_num = Xnew_partition_num.reshape((-1, 1))
        Xnew_replicate_num = Xnew_replicate_num.reshape((-1, 1))

        Xnew = np.concatenate((Xnew, Xnew_replicate_num), axis=1)
        return Xnew, Xnew_partition_num, Xnew_replicate_num


    def __full_splitted_prediction(self, data, time_pred, gaussian_noise):
        X = data[["X", "r"]].values
        y = data[["y"]].values
        mean_f, var_f, mean_r, var_r = self.__exact_prediction_rep(X, y, time_pred, self.__pair_models.kern
                                                                   , gaussian_noise, self.__pair_partition)

        lower_interval_f, upper_interval_f = self.__exact_prediction_quantiles(var_f)
        lower_interval_f = lower_interval_f.flatten()
        upper_interval_f = upper_interval_f.flatten()

        lower_interval_r, upper_interval_r = self.__exact_prediction_quantiles(var_r)
        lower_interval_r = lower_interval_r.flatten()
        upper_interval_r = upper_interval_r.flatten()

        return mean_f, var_f, mean_r, var_r, lower_interval_f, upper_interval_f, lower_interval_r, upper_interval_r


    @abstractmethod
    def plot(self):
        pass

    def __generate_partition_number(self, partition):
        partition_num = []
        for i, subset in enumerate(list(partition)):
            condition_no = [self.__conditions_dict[condition] for condition in subset]
            subset_data = self.__gene_data[self.__gene_data['c'].isin(condition_no)]
            tmp = np.repeat(i + 1, len(subset_data))
            #tmp = np.repeat(i + 1, len(self.__timepoints) * self.__n_replicates * len(subset))

            #print(tmp.shape)
            #print(len(self.__timepoints), self.__n_replicates, len(subset))
            partition_num.append(tmp)
        partition_num = np.concatenate(partition_num)
        return partition_num


    def plot_data(self, timewarping=False, logexpr=False):
        data, _, _ = self.get_conditions_data(self.__conditions)
        partition_num = self.__generate_partition_number([[c] for c in self.__conditions])

        replicate_idx = pd.unique(self.__gene_data['r']).astype(int)

        data['s'] = partition_num
        #data, mu, sigma = self.normalize_gene_data(data)
        #print(mu)
        data = data.sort_values(["s", "r"])

        X = data[["X", "r"]]
        y = data[["y"]]

        #print(data)
        plt.figure(figsize=(22, 8))
        plt.subplot(121)
        for i, subset in enumerate(self.__conditions):

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
                        marker=self.__markers[r_idx],
                        c=self.__colors[i],
                        label=str(subset)
                    )
                else:
                    plt.scatter(
                        time_points,
                        data_points,
                        marker=self.__markers[r_idx],
                        c=self.__colors[i]
                    )
            #plt.grid()

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
            for i, subset in enumerate(self.__conditions):
                # plt.subplot(len(self.conditions),1, i+1)
                subset_data = data[data['s'] == (i + 1)]  # self.get_conditions_data(subset)
                for r in range(self.__n_replicates):
                    r_data = subset_data[subset_data["r"] == (r + 1)]
                    data_points = np.exp(r_data["y"].values) - 1
                    if r == 0:
                        plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r], c=self.__colors[i],
                                    label=str(subset))
                    else:
                        plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r], c=self.__colors[i])
                # plt.grid()

            # plt.tight_layout()
            plt.xlabel("Time")
            #plt.ylabel("Gene read count")
            plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            plt.grid()
        plt.tight_layout()
        plt.show()


    def __get_conditions_dict(self, conditions):
        conditions_dict = {}
        for i, c in enumerate(conditions):
            conditions_dict[c] = i + 1

        return conditions_dict


    def __augment_gene_data(self, gene_data):
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
            c.append(self.__conditions_dict[condition])

        X = np.array(X).reshape((-1,1))
        r = np.array(r).reshape((-1,1))
        c = np.array(c).reshape((-1, 1))
        y = gene_data.values.reshape((-1,1))
        df = pd.DataFrame(np.concatenate((X, c, r, y), axis=1), columns=["X", "c", "r", "y"]).sort_values(["c", "X"])
        return df


    def __get_model_file_name(self, subset, pairing=False):
        name = self.__models_folder + "/" + self.__gene_name + "/"

        if not os.path.exists(name):
            os.makedirs(name)

        for condition in subset:
            name += condition
        if pairing:
            name += "_pairing"
        name += ".pkl"
        return name


    def __delete_models(self):
        for the_file in os.listdir(self.__models_folder):
            file_path = os.path.join(self.__models_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

