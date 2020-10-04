import itertools
import os
import pickle
import shutil
import GPy
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-poster')
matplotlib.rc('axes', titlesize=24)
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

from BaseKernel import BaseKernel

import warnings
warnings.filterwarnings("ignore")

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


class DiffConditionModel:
    def __init__(self, gene_data, gene_name, timepoints, conditions, n_replicates
                , models_folder="./models", hyperparams_iter=3):
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

        self.__timewarping=False
        self.__pairing=False
        self.__replicate_eff=False
        self.__use_stored=False
        self.__use_gpu=False
        
        self.__base_mll = None
        self.__base_partition = None
        self.__base_models = None
        self.__pair_mll = None
        self.__pair_models = None
        self.__pair_partition = None

        self.base_partitions_ranking = None
        self.pair_partitions_ranking = None

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

        self.prior_distribution_f_ls = GPy.priors.LogGaussian(0.5, 0.5)
        self.prior_distribution_g_ls = GPy.priors.LogGaussian(0, 0.5)
        self.prior_distribution_g_var = GPy.priors.Exponential(5)
        # prior_distribution_f_var = GPy.priors.LogGaussian(0.2, 0.5)

    def get_base_model(self):
        return self.__base_models
    def get_pair_model(self):
        return self.__pair_models

    def get_base_partition(self):
        return self.__base_partition
    def get_pair_partition(self):
        return self.__pair_partition

    def get_base_score(self):
        return  self.__base_mll
    def get_pair_score(self):
        return  self.__pair_mll

    def get_gene_data(self):
        return self.__gene_data

    def fit(self, timewarping=True, pairing=False, replicate_eff=False, use_stored=False, use_gpu=False
            , verbose=True, full_pairing=False, k_prior = False):
        self.__timewarping, self.__pairing, self.__replicate_eff = timewarping, pairing, replicate_eff
        self.__use_stored, self.__use_gpu = use_stored, use_gpu

        if use_stored == False:
            # remove all the already trained model
            self.__delete_models()

        self.fit_base_model(verbose=verbose, k_prior=k_prior)

        if pairing:
            self.fit_pairing(verbose=verbose, full_pairing=full_pairing, k_prior=k_prior)

    def fit_base_model(self, verbose=True, k_prior=False):
        partition_models = []
        #for p in tqdm(self.__partitions, disable=(not verbose)):
        for p in tqdm(self.__partitions):
            mll_p = []
            models = []
            for subset in p:
                name = self.__get_model_file_name(subset)
                if os.path.isfile(name):
                    gp, score = load_gp_model(name)
                else:
                    # train a GP on that set of conditions
                    kernel = self.__generate_base_kernel()
                    gp, score = self.gp_base_fit(kernel, subset, k_prior=k_prior)
                    save_gp_model((gp, score), name)
                mll_p.append(score)
                models.append(gp)
            mll = np.sum(mll_p)
            partition_models.append([p, mll])

            if self.__base_mll == None or mll > self.__base_mll:
                self.__base_mll = mll
                self.__base_models = models
                self.__base_partition = p
        self.base_partitions_ranking = pd.DataFrame(partition_models, columns=["p", "score"]).sort_values("score", ascending=False)

    def fit_pairing(self, k_prior=False, verbose=True, full_pairing=False):
        print("Learning pairing effect...")
        #import time
        #start = time.time()
        if full_pairing:
            #self.fit_pairing_multithread(k_prior=k_prior, verbose=k_prior, full_pairing=full_pairing, centered_pairing_kern=centered_pairing_kern)
            partition_models_pair = []
            #for p in tqdm(self.__partitions, disable=(not verbose)):
            for p in tqdm(self.__partitions):
                models, mll = self.gp_pairing_fit(p, k_prior=k_prior)
                partition_models_pair.append([p, mll])

                if self.__pair_mll == None or mll > self.__pair_mll:
                    self.__pair_mll = mll
                    self.__pair_models = models
                    self.__pair_partition = p

            self.pair_partitions_ranking = pd.DataFrame(partition_models_pair, columns=["p", "score_pairing"]).sort_values("score_pairing", ascending=False)

        else:
            models, mll = self.gp_pairing_fit(self.__base_partition, k_prior=k_prior)
            self.__pair_models = models
            self.__pair_partition = self.__base_partition
        #stop = time.time()
        print("Done")
        #print(stop-start)

    def gp_base_fit(self, kernel, subset, k_prior=False):
        data, _, _ = self.get_conditions_data(subset)#, normalize=True)

        if self.__replicate_eff:
            X_y = data[["X", "y", "r"]].sort_values(["r"])
            X = X_y[["X", "r"]]
            y = X_y[["y"]]
        else:
            X = data[["X"]]
            y = data[["y"]]

        if k_prior:
            kernel.lengthscale.set_prior(self.prior_distribution_f_ls, warning=False)

        gp, score = self.__gp_unit(X, y, kernel, normalizer=True)

        return gp, score

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
    '''
    def pairing_data(self, partition):
        data, _, _ = self.get_conditions_data(self.__conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in partition:
            for condition in subset:
                new_data = data[data['c'] == self.__conditions_dict[condition]]
                mu, sigma = new_data['y'].mean(), new_data['y'].std()
                new_data['y'] = (new_data['y'] - mu)/sigma
                ordered_data = ordered_data.append(new_data)
        data = ordered_data

        partition_num = self.__generate_partition_number(partition)

        data['s'] = partition_num
        #data, mu, sigma = self.__normalize_gene_data(data)
        data = data.sort_values(["s", "r"])

        return data, mu, sigma
    '''
    def pairing_data(self, partition):
        data, _, _ = self.get_conditions_data(self.__conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in partition:
            for condition in subset:
                new_data = data[data['c'] == self.__conditions_dict[condition]]
                # to be commented if normalization is done on the entire data
                mu, sigma = new_data['y'].mean(), new_data['y'].std()
                new_data['y'] = (new_data['y'] - mu)/sigma
                # -----------------------------------------------------------
                ordered_data = ordered_data.append(new_data)
        data = ordered_data

        partition_num = self.__generate_partition_number(partition)

        data['s'] = partition_num
        # to be commented if normalization is done response by response
        #data, mu, sigma = self.__normalize_gene_data(data)
        # -------------------------------------------------------------
        data = data.sort_values(["s", "r"])

        return data, mu, sigma

    def __generate_base_kernel(self):
        rbf_f = GPy.kern.RBF(input_dim=1, useGPU=self.__use_gpu, name="f")

        if self.__replicate_eff:
            kern_g = GPy.kern.RBF(input_dim=1, useGPU=self.__use_gpu, name='g')
            kernel = GPy.kern.Hierarchical(kernels=[rbf_f, kern_g])
        else:
            kernel = rbf_f

        return kernel

    def __generate_base_kernel_full(self, data, partition, k_prior):
        partition_num = self.__generate_partition_number(partition)
        partition_mat_list = get_partition_mat(partition_num)
        rep_mat_list, specular_rep_mat_list = get_partition_mat(data['r'].values, get_specular=True)

        rbf_f = []
        for i in range(len(partition)):
            k = GPy.kern.RBF(input_dim=1, useGPU=self.__use_gpu, name="f" + str(i))
            if k_prior:
                k.lengthscale.set_prior(self.prior_distribution_f_ls, warning=False)
            #k.variance.set_prior(prior_distribution_f_var)
            rbf_f.append(k)

        kernel = BaseKernel(
            rbf_f, 
            partition_num, 
            partition_mat_list,
        )
        
        return kernel

    def gp_pairing_fit(self, partition, k_prior=False):
        data, _, _ = self.pairing_data(partition)

        kernel = self.__generate_base_kernel_full(
            data,
            partition,
            k_prior,
        )

        X = data[['X', 'r']].values
        y = data[['y']].values

        gp, score = self.__gp_unit(X, y, kernel)

        #score = self.log_marginal_likelihood(gp, X, y, partition)

        return gp, score

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

        K_XnewX = full_kern.K(
            X,
            X2=Xnew,
            X2_subset_num=Xnew_partition_num,
        )
        K_XnewXnew = full_kern.K(
            Xnew,
            X2=Xnew,
            X2_subset_num=Xnew_partition_num,
            X_subset_num=Xnew_partition_num,
        )
        mean = K_XnewX.T.dot(alpha).dot(y)
        var = K_XnewXnew - K_XnewX.T.dot(alpha).dot(K_XnewX)

        return mean, np.diag(var)


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

    def plot(self, title=None):
        time_pred = self.__get_time_pred()
        time_predN = time_pred.reshape((-1, 1))

        replicate_idx = pd.unique(self.__gene_data['r']).astype(int)

        plt.figure(figsize=(16, 10))
        for i, subset in enumerate(self.__base_partition):
            label = get_label(subset)

            if self.__replicate_eff:
                y_pred_mean_f, _ = self.__base_models[i].predict(time_predN, kern=self.__base_models[i].kern.f)
                lower_interval, upper_interval = self.__base_models[i].predict_quantiles(time_predN,
                                                                                         kern=self.__base_models[i].kern.f)
            else:
                y_pred_mean_f, y_pred_var_f = self.__base_models[i].predict(time_predN)
                lower_interval, upper_interval = self.__base_models[i].predict_quantiles(time_predN)

            y_pred_mean_f = y_pred_mean_f.reshape(len(y_pred_mean_f))

            upper_interval = upper_interval.flatten()
            lower_interval = lower_interval.flatten()

            #upper_interval = np.exp(upper_interval)-1
            #lower_interval = np.exp(lower_interval)-1

            plt.plot(time_pred, upper_interval, color=self.__colors[i], linestyle="--")
            plt.plot(time_pred, lower_interval, color=self.__colors[i], linestyle="--")
            plt.fill_between(time_pred, lower_interval, upper_interval, color=self.__colors[i], alpha=0.05)

            if self.__replicate_eff:
                f, g = self.__base_models[i].kern.f, self.__base_models[i].kern.g
                var_f = f.variance[0]
                ls_f = f.lengthscale[0]
                var_g = g.variance[0]
                ls_g = g.lengthscale[0]
                label += "\n(var1=%.2f, ls1=%.2f)+(var2=%.2f, ls2=%.2f)" % (var_f, ls_f, var_g, ls_g)
            else:
                variance = self.__base_models[i].kern.to_dict()['variance'][0]
                lengthscale = self.__base_models[i].kern.to_dict()['lengthscale'][0]
                label += '\n(alpha=%.2f , l=%.2f)' % (variance, lengthscale)

            #y_pred_mean_f = np.exp(y_pred_mean_f)-1
            #print("about to plot")
            #print(time_pred)
            #print(y_pred_mean_f)
            plt.plot(time_pred, y_pred_mean_f, self.__colors[i], label=label)

            data, _, _ = self.get_conditions_data(subset)  # , normalize=True)
            for r_idx, r in enumerate(replicate_idx):
                r_data = data[data["r"] == r]
                plt.scatter(r_data["X"].values, r_data["y"].values, marker=self.__markers[r_idx], c=self.__colors[i],
                            label="replicate " + str(r))

        if self.__timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel("gene expression level", fontsize=22)  # [log(y+1)]")

        mll = "Marginal likelihood = %.2f" % self.__base_mll

        if title is None:
            plt.title("" + self.__gene_name + "")
        else:
            plt.title(title)
        plt.legend(prop={'size': 24})#prop={'size': 14}, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid()

        #plt.savefig(self.__gene_name + "_base_model.jpg", dpi=350, format="jpg")

        plt.show()

        if self.__replicate_eff:
            self.__plot_replicate_eff(time_pred, y_pred_mean_f)

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
        plt.figure(figsize=(16, 10))
        #plt.subplot(121)
        for i, subset in enumerate(self.__conditions):

            #plt.subplot(len(self.conditions),1, i+1)
            subset_data = data[data['s'] == (i + 1)]  # self.get_conditions_data(subset)
            for r_idx, r in enumerate(replicate_idx):
                r_data = subset_data[subset_data["r"] == r]
                if logexpr:
                    data_points = r_data["y"].values
                else:
                    data_points = np.exp(r_data["y"].values)-1
                if r_idx==0:
                    plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r_idx], c=self.__colors[i],
                                label=str(subset))
                else:
                    plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r_idx], c=self.__colors[i])
            #plt.grid()

            #plt.tight_layout()

            plt.xlabel("Time (hours)")
            plt.ylabel("Gene read count")
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

    def plot_pairing(self, noiseless=False, title=None):
        time_pred = self.__get_time_pred()
        time_predN = time_pred.reshape((-1, 1))
        print("Marginal log likelihood = " + str(self.__pair_models.log_likelihood()))

        if noiseless:
            gaussian_noise = 0
        else:
            gaussian_noise = self.__pair_models.Gaussian_noise[0]

        data, _, _ = self.get_conditions_data(self.__conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in self.__pair_partition:
            for condition in subset:
                ordered_data = ordered_data.append(data[data['c'] == self.__conditions_dict[condition]])
        data = ordered_data

        partition_num = self.__generate_partition_number(self.__pair_partition)

        data['s'] = partition_num
        data, mu, sigma = self.__normalize_gene_data(data)

        data = data.sort_values(["s", "r"])

        X = data[["X", "r"]]
        y = data[["y"]]

        mean, var, lower_interval, upper_interval = self.__full_predictions(data, time_pred, gaussian_noise)
        self.__plot_base(
            data,
            mu,
            sigma,
            time_pred,
            mean,
            var,
            lower_interval,
            upper_interval,
            title=title
        )
        self.__plot_kernels(sigma, X.values)

    def __plot_base(self, data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f, title=None):
        plt.figure(figsize=(16,10))
        replicate_idx = pd.unique(data['r']).astype(int)
        for i in range(int(len(self.__pair_partition) * self.__n_replicates)):
            if (i % self.__n_replicates) == 0:
                a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))

                mean_f_subset = mean_f[a:b].reshape(b - a)

                color_idx = i // self.__n_replicates
                c = self.__colors[color_idx]

                plt.plot(time_pred, (mean_f_subset*sigma) + mu, color=c, label=get_label(self.__pair_partition[color_idx]))

                lower = mean_f_subset + lower_interval_f[a:b]
                upper = mean_f_subset + upper_interval_f[a:b]

                lower = (lower * sigma) + mu
                upper = (upper * sigma) + mu

                plt.plot(time_pred, lower, color=c, linestyle="--")
                plt.plot(time_pred, upper, color=c, linestyle="--")
                plt.fill_between(time_pred, lower, upper, color=c, alpha=0.05)

        for i, subset in enumerate(self.__pair_partition):
            subset_data = data[data['s'] == (i + 1)]  # self.get_conditions_data(subset)
            for r_idx, r in enumerate(replicate_idx):
                r_data = subset_data[subset_data["r"] == r]# (r + 1)]
                data_points = (r_data["y"].values * sigma) + mu
                plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r_idx], c=self.__colors[i],
                            label="replicate " + str(r))

        if title is None:
            plt.title("" + self.__gene_name + "")
        else:
            plt.title(title)
        if self.__timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel("gene expression level", fontsize=22)
        #plt.legend(prop={'size': 24})
        plt.legend(prop={'size': 24}, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid()
        #plt.ylim([1.3,2.8])
        #plt.savefig(self.__gene_name + "_pair_model.svg")
        plt.savefig(self.__gene_name + "_pair_model.jpg", dpi=350, format="jpg")
        plt.show()

    def __plot_base_separated(self, data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f):
        for i in range(int(len(self.__pair_partition) * self.__n_replicates)):

            if (i % self.__n_replicates) == 0:
                plt.figure(figsize=(16, 10))
                a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))

                mean_f_subset = mean_f[a:b].reshape(b - a)

                color_idx = i // self.__n_replicates
                c = self.__colors[color_idx]

                plt.plot(time_pred, (mean_f_subset*sigma) + mu, color=c, label=get_label(self.__pair_partition[color_idx]))

                lower = mean_f_subset + lower_interval_f[a:b]
                upper = mean_f_subset + upper_interval_f[a:b]

                lower = (lower * sigma) + mu
                upper = (upper * sigma) + mu

                plt.plot(time_pred, lower, color=c, linestyle="--")
                plt.plot(time_pred, upper, color=c, linestyle="--")
                plt.fill_between(time_pred, lower, upper, color=c, alpha=0.05)

                subset_idx = i // self.__n_replicates
                subset_data = data[data['s'] == (subset_idx + 1)]  # self.get_conditions_data(self.best_partition_pair[subset_idx])
                for r in range(self.__n_replicates):
                    r_data = subset_data[subset_data["r"] == (r + 1)]
                    c_data = r_data[r_data["c"] == (subset_idx + 1)]
                    data_points = (c_data["y"].values * sigma) + mu
                    plt.scatter(c_data["X"].values, data_points, marker=self.__markers[r],
                                c=self.__colors[subset_idx],
                                label="replicate " + str(r + 1))
                plt.title("$f_s$")
                plt.legend()
                plt.grid()
                plt.show()

    def __plot_kernels(self, sigma, X = None):
        if np.any(X != None):
            X = X
        else:
            data, _, _ = self.get_conditions_data(self.__conditions)
            partition_num = self.__generate_partition_number(self.__pair_partition)

            data['s'] = partition_num
            data = data.sort_values(["s", "r"])
            X = data[["X", "r"]].values

        plt.figure(figsize=(5, 5))
        plt.imshow(self.__pair_models.kern.K(X) * (sigma ** 2), cmap="hot")
        plt.title("$Kp$")

        plt.colorbar()
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

