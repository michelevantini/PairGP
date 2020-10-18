# BASE IMPORT
import os
import pickle
import shutil

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

from DiffConditionModel import DiffConditionModel

class PairingEffectModel(DiffConditionModel):
    def __init__(
        self,
        gene_data,
        gene_name,
        timepoints,
        conditions,
        n_replicates,
        models_folder="./models",
        hyperparams_iter=5
    ):
        super().__init__()

        self.prior_distribution_f_ls = GPy.priors.LogGaussian(0.5,0.5)
        self.prior_distribution_g_ls = GPy.priors.LogGaussian(0,0.5)
        self.prior_distribution_g_var = GPy.priors.Exponential(2)
        # prior_distribution_f_var = GPy.priors.LogGaussian(0.2, 0.5)

    def fit(
        self,
        timewarping=True,
        k_prior = False,
        use_stored=False,
        use_gpu=False,
        verbose=True, 
    ):
        self.__timewarping = timewarping
        self.__use_stored = use_stored
        self.__use_gpu = use_gpu

        if use_stored == False:
            # remove all the already trained model
            self.__delete_models()

        if verbose:
            import time
            start = time.time()

        partition_models_pair = []
        for p in tqdm(self.__partitions, disable=(not verbose)):
            models, mll = self.gp_pairing_fit(
                p, 
                k_prior=k_prior,
            )
            partition_models_pair.append([p, mll])

            if self.__pair_mll == None or mll > self.__pair_mll:
                self.__pair_mll = mll
                self.__pair_models = models
                self.__pair_partition = p

        self.pair_partitions_ranking = pd.DataFrame(
            partition_models_pair,
            columns=["p", "score_pairing"]
        ).sort_values("score_pairing", ascending=False)

        if verbose:
            print("Done")
            stop = time.time()
            print("Model fitting time:", stop-start)

    def pairing_data(self, partition):
        data, _, _ = self.get_conditions_data(self.__conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in partition:
            for condition in subset:
                ordered_data = ordered_data.append(data[data['c'] == self.__conditions_dict[condition]])
        data = ordered_data

        partition_num = self.__generate_partition_number(partition)

        data['s'] = partition_num
        data, mu, sigma = self.__normalize_gene_data(data)
        data = data.sort_values(["s", "r"])

        return data, mu, sigma

    def __generate_pairing_kernel(self, data, partition, k_prior, centered_pairing_kern, pairing_eff_var_prior=None):
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

        # if single_g_kern:
        rbf_g = GPy.kern.RBF(input_dim=1, useGPU=self.__use_gpu, name='g')
        if k_prior:
            if pairing_eff_var_prior == None:
                rbf_g.variance.set_prior(self.prior_distribution_g_var, warning=False)
            else:
                rbf_g.variance.set_prior(GPy.priors.Exponential(pairing_eff_var_prior), warning=False)
            rbf_g.lengthscale.set_prior(self.prior_distribution_g_ls, warning=False)
        kernel = PairingEffectKernel([rbf_f, rbf_g], partition_num, data['r'].values, partition_mat_list,
                                     rep_mat_list, self.__n_replicates, specular_rep_mat_list=specular_rep_mat_list
                                     , centered_kernel=centered_pairing_kern)
        return kernel

    def gp_pairing_fit(self, partition, k_prior=False
                       , centered_pairing_kern=False, pairing_eff_var_prior=None):
        data, _, _ = self.pairing_data(partition)
        kernel = self.__generate_pairing_kernel(
            data,
            partition,
            k_prior,
            centered_pairing_kern,
            pairing_eff_var_prior = pairing_eff_var_prior,
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

    def __normalize_gene_data(
        self,
        data
    ):
        """Given a pd.DataFrame with a column 'y' containing
        the gene expression data, it returns the pd.DataFrame
        where the columns 'y' is normalized

        Args:
            data (pd.DataFrame): pd.DataFrame to normalize

        Returns:
            pd.DataFrame: normalized pd.DataFrame
        """
        mu, sigma = np.mean(data['y']), np.std(data['y'])
        data['y'] = (data['y']-mu)/sigma

        return data, mu, sigma

    def plot(self, title=None):
        time_pred = self.__get_time_pred()
        time_predN = time_pred.reshape((-1, 1))

        replicate_idx = pd.unique(self.__gene_data['r']).astype(int)

        plt.figure(figsize=(16,7))
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
                #label += "\n(var1=%.2f, ls1=%.2f)+(var2=%.2f, ls2=%.2f)" % (var_f, ls_f, var_g, ls_g)
            else:
                variance = self.__base_models[i].kern.to_dict()['variance'][0]
                lengthscale = self.__base_models[i].kern.to_dict()['lengthscale'][0]
                #label += '\n(alpha=%.2f , l=%.2f)' % (variance, lengthscale)

            #y_pred_mean_f = np.exp(y_pred_mean_f)-1
            #print("about to plot")
            #print(time_pred)
            #print(y_pred_mean_f)
            plt.plot(time_pred, y_pred_mean_f, self.__colors[i], label=label)

            data, _, _ = self.get_conditions_data(subset)  # , normalize=True)
            for r_idx, r in enumerate(replicate_idx):
                r_data = data[data["r"] == r]
                plt.scatter(
                    r_data["X"].values,
                    r_data["y"].values,
                    marker=self.__markers[r_idx],
                    c=self.__colors[i],
                    #label="replicate " + str(r),
                )

        if self.__timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel("gene expression level (log(y+1))", fontsize=22)  # [log(y+1)]")

        mll = "Marginal likelihood = %.2f" % self.__base_mll

        if title is None:
            plt.title("" + self.__gene_name + "")
        else:
            plt.title(title)
        plt.legend(prop={'size': 24})
        #plt.legend(prop={'size': 24}, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid()
        plt.tight_layout()
        plt.savefig("gene_figures/" + self.__gene_name + "_base_model.jpg", format="jpg")

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

        #self.__plot_base_plus_pairing(data, mu, sigma, time_pred, mean, lower_interval, upper_interval)
        #self.__plot_base_plus_pairing_allr(data, mu, sigma, time_pred, mean, lower_interval, upper_interval)

        mean_f, var_f, mean_r, var_r, lower_interval_f, upper_interval_f, lower_interval_r, upper_interval_r = self.__full_splitted_prediction(data, time_pred, gaussian_noise)

        self.__plot_base(data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f, title=title)
        #self.plot_base_separated(data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f)

        self.__plot_pairing_effect(data, mu, sigma, time_pred, mean_r, mean_f, lower_interval_r, upper_interval_r)
        #K, K_rep = self.__plot_kernels(sigma, X.values)
        #return K, K_rep

    def __plot_base(self, data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f, title=None):
        plt.figure(figsize=(16,7))
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
                plt.scatter(
                    r_data["X"].values,
                    data_points,
                    marker=self.__markers[r_idx],
                    c=self.__colors[i],
                    #label="replicate " + str(r),
                )

        if title is None:
            plt.title("" + self.__gene_name + "")
        else:
            plt.title(title)
        if self.__timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel("gene expression level", fontsize=22)
        plt.legend(prop={'size': 24})
        plt.grid()
        #plt.ylim([1.3,2.8])
        #plt.savefig("gene_figures/" + self.__gene_name + "_pair_model.svg")
        #plt.tight_layout()
        plt.tight_layout(rect=[0.015,0,1,1])
        plt.savefig("gene_figures/" + self.__gene_name + "_pair_model.jpg", format="jpg")
        plt.show()

    def __plot_pairing_effect(self, data, mu, sigma, time_pred, mean_r, mean_f, lower_interval_r, upper_interval_r):
        plt.figure(figsize=(16,7))

        #mean = ((mean_f+mean_r)*sigma)+mu
        #mean_f_ = (mean_f*sigma)+mu
        #mean_r = mean - mean_f_
        mean_r_rescaled = mean_r*sigma

        for i in range(int(self.__n_replicates)):
            a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))

            mean_r_subset = mean_r[a:b].reshape(b - a)

            color_idx = i % self.__n_replicates
            c = self.__repcolors[color_idx]

            mean_r_subset_denormalized = mean_r_subset*sigma
            plt.plot(time_pred, mean_r_subset_denormalized, color=c)
            plt.scatter(time_pred[::10], mean_r_subset_denormalized[::10], color=c, marker=self.__markers[i]
                        , label="replicate " + str(color_idx + 1))

            lower = mean_r_subset + lower_interval_r[a:b]
            upper = mean_r_subset + upper_interval_r[a:b]

            plt.plot(time_pred, lower*sigma, color=c, linestyle="--")
            plt.plot(time_pred, upper*sigma, color=c, linestyle="--")
            plt.fill_between(time_pred, lower*sigma, upper*sigma, color=c, alpha=0.05)

        plt.title("Pairing effect")
        #plt.legend(prop={'size': 24})
        plt.legend(prop={'size': 24}, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        '''
        plt.legend(
            prop={'size': 24},
            bbox_to_anchor=(0.02, 0.98),
            loc='upper left',
            ncol=3,
            borderaxespad=0.
        )
        '''
        if self.__timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.grid()
        #plt.yticks([-0.5,0,0.5,1])
        plt.tight_layout()
        plt.savefig("gene_figures/" + self.__gene_name + "_pair_eff.jpg", format="jpg")
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

        plt.figure(figsize=(16, 5))
        plt.subplot(131)
        plt.imshow(self.__pair_models.kern.K(X) * (sigma ** 2), cmap="hot")
        plt.title("$K_r+K_p$")

        plt.colorbar()

        K, K_rep = self.__pair_models.kern.compute_K_Krep(X)
        plt.subplot(132)
        plt.imshow(K*(sigma**2), cmap="hot")
        plt.title("$K_r$")

        plt.colorbar()

        plt.subplot(133)
        plt.imshow(K_rep*(sigma**2), cmap="hot")

        plt.colorbar()
        plt.title("$K_p$")
        plt.tight_layout()
        plt.savefig("gene_figures/" + self.__gene_name + "_k.jpg", format="jpg")
        plt.show()

        return K,K_rep

    def __plot_replicate_eff(self, time_pred, y_pred_mean_f):
        time_predN = time_pred.reshape((-1, 1))

        for i, subset in enumerate(self.__base_partition):
            label = get_label(subset)

            plt.figure(figsize=(16,10))
            for r in range(self.__n_replicates):
                time_predN_g = np.hstack((time_predN, np.repeat(r, 100).reshape((100, 1))))
                y_pred_mean_g, _ = self.__base_models[i].predict(time_predN_g)
                y_pred_mean_g = y_pred_mean_g.reshape(len(y_pred_mean_g))

                lower_interval, upper_interval = self.__base_models[i].predict_quantiles(time_predN_g)
                upper_interval = upper_interval.flatten() - y_pred_mean_f
                lower_interval = lower_interval.flatten() - y_pred_mean_f
                plt.plot(time_pred, upper_interval, self.__colors[r] + "--")

                plt.plot(time_pred, lower_interval, self.__colors[r] + "--")
                plt.fill_between(time_pred, lower_interval, upper_interval, color=self.__colors[r], alpha=0.05)

                y_pred_mean_g = y_pred_mean_g - y_pred_mean_f
                plt.plot(time_pred, y_pred_mean_g, self.__colors[r], label="replicate " + str(r))

            plt.xlabel("Time", fontsize=16)
            plt.ylabel("Gene expression level [log(y+1)]", fontsize=16)

            plt.title(self.__gene_name + " - " + label, fontsize=18)
            plt.legend(prop={'size': 14})
            plt.grid()
        plt.show()

    def __plot_base_plus_pairing(self, data, mu, sigma, time_pred, mean, lower_interval, upper_interval):
        for i in range(int(len(self.__pair_partition) * self.__n_replicates)):
            plt.figure(figsize=(16, 6))

            a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))
            mean_subset = mean[a:b].reshape(b - a)

            color_idx = i % self.__n_replicates
            subcolor_idx = i // self.__n_replicates
            r = color_idx
            c = self.__colors[subcolor_idx]

            plt.plot(time_pred, (mean_subset*sigma)+mu, color=c)

            lower = mean_subset + lower_interval[a:b]
            upper = mean_subset + upper_interval[a:b]

            lower = (lower * sigma) + mu
            upper = (upper * sigma) + mu

            plt.plot(time_pred, lower, color=c, linestyle="--")
            plt.plot(time_pred, upper, color=c, linestyle="--")
            plt.fill_between(time_pred, lower, upper, color=c, alpha=0.05)

            j, subset = subcolor_idx, self.__pair_partition[subcolor_idx]
            subset_data = data[data['s'] == (j + 1)]  # self.get_conditions_data(subset)
            r_data = subset_data[subset_data["r"] == (r + 1)]
            data_points = (r_data["y"].values * sigma) + mu
            plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r], c=c,
                        label="replicate " + str(r + 1))
            if r == 0:
                plt.title(get_label(subset) + " ($f_s + f_r$)")

            plt.legend()
            plt.grid()

            plt.show()

    def __plot_base_plus_pairing_allr(self, data, mu, sigma, time_pred, mean, lower_interval, upper_interval):
        subset_idx = 0
        replicate_idx = pd.unique(data['r']).astype(int)

        plt.figure(figsize=(16, 10))
        for i in range(int(len(self.__pair_partition) * self.__n_replicates)):
            if (i != 0) and ((i % self.__n_replicates) == 0):
                plt.figure(figsize=(16, 10))
            a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))
            mean_subset = mean[a:b].reshape(b - a)

            color_idx = i % self.__n_replicates
            subcolor_idx = i // self.__n_replicates
            subcol = self.__subcolors[subcolor_idx]
            r = color_idx
            c = self.__colors[subcolor_idx]

            plt.plot(time_pred, (mean_subset * sigma) + mu, color=subcol[r])

            lower = mean_subset + lower_interval[a:b]
            upper = mean_subset + upper_interval[a:b]

            lower = (lower * sigma) + mu
            upper = (upper * sigma) + mu

            plt.plot(time_pred, lower, color=subcol[r], linestyle="--")
            plt.plot(time_pred, upper, color=subcol[r], linestyle="--")
            plt.fill_between(time_pred, lower, upper, color=subcol[r], alpha=0.05)

            if (i != 0) and (((i + 1) % self.__n_replicates) == 0):
                j, subset = subcolor_idx, self.__pair_partition[subcolor_idx]
                if j == subset_idx:
                    subcol = self.__subcolors[j]
                subset_data = data[data['s'] == (j + 1)]  # self.get_conditions_data(subset)
                for c_idx, r in enumerate(replicate_idx):
                    c = subcol[c_idx]
                    r_data = subset_data[subset_data["r"] == r]#(r + 1)]
                    data_points = (r_data["y"].values*sigma)+mu
                    plt.scatter(r_data["X"].values, data_points, marker=self.__markers[r], c=c,
                                label="replicate " + str(r))
                    if r == 0:
                        plt.title(get_label(subset) + " ($f_s + f_r$)")
                subset_idx += 1
                plt.legend()
                plt.grid()

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

