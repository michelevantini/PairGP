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

class DiffConditionModel:
    def __init__(self, gene_data, gene_name, timepoints, conditions, n_replicates
                , models_folder="./models", hyperparams_iter=5):
        self.__conditions = conditions
        self.__partitions = list(partition(conditions))
        self.__conditions_dict = self.__get_conditions_dict(conditions)
        self.__gene_data = self.__augment_gene_data(gene_data)

        self.__gene_name = gene_name
        self.__timepoints = timepoints

        self.__hyperparams_iter = hyperparams_iter
        self.__n_replicates = n_replicates

        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        self.__models_folder = models_folder

        self.__use_stored=False
        self.__use_gpu=False
        
        self.__mll = None
        self.__partition = None
        self.__models = None

        self.partitions_ranking = None

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
        self.__TIME_PRED_LEN = 100

    def get_models(self):
        return self.__models
    
    def get_partition(self):
        return self.__partition
    
    def get_score(self):
        return  self.__mll
    
    def get_gene_data(self):
        return self.__gene_data

    @abstractmethod
    def fit(self):
        pass
        
    def __gp_unit(self, X, y, kernel, normalizer=False):
        gp = GPy.models.GPRegression(X, y, kernel, normalizer=normalizer)
        gp.optimize_restarts(num_restarts=self.__hyperparams_iter, verbose=False)
        score = gp.log_likelihood()
        return gp, score

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

    def __normalize_gene_data(self, data):
        mu, sigma = np.mean(data['y']), np.std(data['y'])
        data['y'] = (data['y']-mu)/sigma

        return data, mu, sigma

    def get_conditions_data(self, subset, normalize=False):
        '''  
        Filter data to return only the data related to subset
        This function also applies the proper transformation to the data
        :param subset: subset of conditions to filter
        :param timewarping: whether to use timewarping or not 
        :param a,b: beta cdf parameter, to be set only if beta cdf has been used
        :return: X: time
                 y: log normalized data
                 mu: mean of the log transformed data
                 sigma: standard deviation of the transformed data
                 columns: columns selected from data
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

    def __get_time_pred(self):
        if self.__timewarping:
            #return np.linspace(time_warping(self.timepoints[0]), time_warping(self.timepoints[-1]), self.TIME_PRED_LEN)
            #tmp = time_warping(self.__timepoints[0])
            #print(tmp)
            #tmp = np.linspace(time_warping(self.__timepoints[0]), time_warping(self.__timepoints[-1]), self.__TIME_PRED_LEN)
            #print(tmp)
            return np.linspace(time_warping(self.__timepoints[0]), time_warping(self.__timepoints[-1]), self.__TIME_PRED_LEN)
        else:
            return np.linspace(self.__timepoints[0], self.__timepoints[-1], self.__TIME_PRED_LEN)

    def __exact_prediction_full(self, X, y, time_pred, full_kern, noise, partition):
        Xnew, Xnew_partition_num, Xnew_replicate_num = self.__generate_Xnew(time_pred, partition)
        K_XX = full_kern.K(X)

        if noise==0:
            alpha=np.linalg.inv(K_XX + 1e-5 * np.eye(X.shape[0]))
        else:
            alpha = np.linalg.inv(K_XX + noise * np.eye(X.shape[0]))

        K_XnewX = full_kern.K(X, X2=Xnew, X2_subset_num=Xnew_partition_num, X2_rep_num=Xnew_replicate_num)
        K_XnewXnew = full_kern.K(Xnew, X2=Xnew, X2_subset_num=Xnew_partition_num, X2_rep_num=Xnew_replicate_num,
                                 X_subset_num=Xnew_partition_num, X_rep_num=Xnew_replicate_num)
        mean = K_XnewX.T.dot(alpha).dot(y)
        var = K_XnewXnew - K_XnewX.T.dot(alpha).dot(K_XnewX)

        return mean, np.diag(var)

    def __exact_prediction_rep(self, X, y, time_pred, full_kern, noise, partition):
        Xnew, Xnew_partition_num, Xnew_replicate_num = self.__generate_Xnew(time_pred, partition)
        K_XX , Krep_XX = full_kern.compute_K_Krep(X)

        K_XnewX, Krep_XnewX = full_kern.compute_K_Krep(X, X2=Xnew, X2_subset_num=Xnew_partition_num, X2_rep_num=Xnew_replicate_num)
        K_XnewXnew, Krep_XnewXnew = full_kern.compute_K_Krep(Xnew, X2=Xnew, X2_subset_num=Xnew_partition_num, X2_rep_num=Xnew_replicate_num,
                                 X_subset_num=Xnew_partition_num, X_rep_num=Xnew_replicate_num)

        if noise==0:
            alpha=np.linalg.inv(K_XX + Krep_XX + 1e-5 * np.eye(X.shape[0]))
        else:
            alpha = np.linalg.inv(K_XX + Krep_XX + noise * np.eye(X.shape[0]))

        mean_f = K_XnewX.T.dot(alpha).dot(y)
        var_f = K_XnewXnew - K_XnewX.T.dot(alpha).dot(K_XnewX)

        mean_r = Krep_XnewX.T.dot(alpha).dot(y)
        var_r = Krep_XnewXnew - Krep_XnewX.T.dot(alpha).dot(Krep_XnewX)

        return mean_f, np.diag(var_f), mean_r, np.diag(var_r)

    def __exact_prediction_quantiles(self, var):
        quantiles = (2.5, 97.5)
        #return [stats.norm.ppf(q/100.)*np.sqrt(var + sigma2) for q in quantiles]
        return [stats.norm.ppf(q / 100.) * np.sqrt(var) for q in quantiles]

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

    def __full_predictions(self, data, time_pred, gaussian_noise):
        X = data[["X", "r"]].values
        y = data[["y"]].values
        mean, var = self.__exact_prediction_full(X, y, time_pred, self.__pair_models.kern
                                                 , gaussian_noise, self.__pair_partition)

        lower_interval, upper_interval = self.__exact_prediction_quantiles(var)
        lower_interval, upper_interval = lower_interval.flatten(), upper_interval.flatten()

        return mean, var, lower_interval, upper_interval

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
            conditions_dict[c] = i +1

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

