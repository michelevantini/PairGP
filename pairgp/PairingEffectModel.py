# BASE IMPORT
import os
import pickle
import shutil
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
from pairgp.utils import (
    get_label,
    get_partition_mat,
    log_plus_one,
    load_gp_model,
    save_gp_model,
    partition,
    time_warping,
)

from pairgp.DiffConditionModel import DiffConditionModel
from pairgp.PairingEffectKernel import PairingEffectKernel

# Remove comments here if you don't
# want to see any wornings
# import warnings
# warnings.filterwarnings("ignore")

# Add some personal settings here for
# visualization purposes
plt.style.use("seaborn-poster")
matplotlib.rc("axes", titlesize=24)
matplotlib.rc("xtick", labelsize=24)
matplotlib.rc("ytick", labelsize=24)


class PairingEffectModel(DiffConditionModel):
    def __init__(self, *args, **kwargs):
        super(PairingEffectModel, self).__init__(*args, **kwargs)

        # we define here the hyperprior distributions for the
        # pairing effect model. Refer to GPy documentation https://gpy.readthedocs.io/en/deploy/GPy.kern.src.html
        # for the underlying formulas.
        # Prior distribution for the lengthscale parameter of
        # the response effect kernel
        self.prior_distribution_f_ls = GPy.priors.LogGaussian(0.5, 0.5)
        # Prior distribution for the lengthscale parameter of
        # the pairing effect kernel
        self.prior_distribution_g_ls = GPy.priors.LogGaussian(0, 0.5)
        # Prior distribution for the variance parameter of
        # the pairing effect kernel
        self.prior_distribution_g_var = GPy.priors.Exponential(2)
        # prior_distribution_f_var = GPy.priors.LogGaussian(0.2, 0.5)

        # these attributes keep the state of the object
        # by referring to the best model obtained through
        # the model selection precess (the fit function)
        self.mll = None
        self.models = None
        self.partition = None

    def fit(
        self,
        timewarping: bool = True,
        k_prior: bool = True,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        """Model selection process for the Pairing effect model
        Each partition of the condition set is assessed and
        the best partition is stored in the class attributes:
            - self.mll: marginal likelihood value
            - self.models: the models that are part of the partition
            - self.partition: the best partition selected

        Args:
            timewarping (bool, optional): Apply log-timewarping on the time points.
                Defaults to True:bool.
            k_prior (bool, optional): Apply the kerenel hyperprior distributions
                presented in the constructor of the class. Defaults to True:bool.
            use_gpu (bool, optional): The Gaussian kernel used in this application
                allow to perform the computations also on GPU. Defaults to False:bool.
            verbose (bool, optional): Used to log more information during the model
                selection process in the terminal. Defaults to True:bool.
        """
        self.timewarping = timewarping
        self.use_gpu = use_gpu

        if verbose:
            import time

            start = time.time()

        # Iteration over all the possible partitions of the condition set
        partition_models_pair = []

        for p in tqdm(self.partitions, disable=(not verbose)):
            models, mll = self.gp_pairing_fit(
                p,
                k_prior=k_prior,
            )
            partition_models_pair.append([p, mll])

            if self.mll == None or mll > self.mll:
                self.mll = mll
                self.models = models
                self.partition = p

        self.partitions_ranking = pd.DataFrame(
            partition_models_pair, columns=["p", "score_pairing"]
        ).sort_values("score_pairing", ascending=False)

        if verbose:
            print("Done")
            stop = time.time()
            print("Model fitting time:", stop - start)

    def pairing_data(
        self,
        partition: List[List[str]],
    ):
        """Specific function to preparate data for the pairing effect model
        Normalization of the data is also applied and the mean and std used
        for the transformation are returned.

        Args:
            partition (List[List[str]]): a partition of the condition set

        Returns:
            (pd.DataFrame, float, float): (transformed data, data mean, data std)
        """
        data, _, _ = self.get_conditions_data(self.conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in partition:
            for condition in subset:
                ordered_data = ordered_data.append(
                    data[data["c"] == self.conditions_dict[condition]]
                )
        data = ordered_data

        partition_num = self.generate_partition_number(partition)

        data["s"] = partition_num
        data, mu, sigma = self.normalize_gene_data(data)
        data = data.sort_values(["s", "r"])

        return data, mu, sigma

    def generate_pairing_kernel(
        self,
        data: pd.DataFrame,
        partition: List[List[str]],
        k_prior: bool,
    ):
        """Generate the Pairing effect model kernels to be used
        to fir the pairing effect model to a specific partition
        of the condition set.

        Args:
            data (pd.DataFrame): gene expression data
            partition (List[List[str]]): a partition of the condition set
            k_prior (bool): to apply the hyper-prior distributions for the
                pairing effect model kernel parameters

        Returns:
            PairingEffectKernel: kernel to be used to fit the pairing effect model
        """

        # some preparation of the data
        partition_num = self.generate_partition_number(partition)
        partition_mat_list = get_partition_mat(partition_num)
        rep_mat_list, specular_rep_mat_list = get_partition_mat(
            data["r"].values, get_specular=True
        )

        # create the response kernel
        # one for each subset in the partition
        rbf_f = []
        for i in range(len(partition)):
            k = GPy.kern.RBF(input_dim=1, useGPU=self.use_gpu, name="f" + str(i))
            if k_prior:
                # apply the hyper-prior distribution
                k.lengthscale.set_prior(self.prior_distribution_f_ls, warning=False)
            # k.variance.set_prior(prior_distribution_f_var)
            rbf_f.append(k)

        # create the pairing effect kernel
        rbf_g = GPy.kern.RBF(input_dim=1, useGPU=self.use_gpu, name="g")
        if k_prior:
            # apply the hyper-prior distributions
            rbf_g.variance.set_prior(self.prior_distribution_g_var, warning=False)
            rbf_g.lengthscale.set_prior(self.prior_distribution_g_ls, warning=False)

        kernel = PairingEffectKernel(
            [rbf_f, rbf_g],
            partition_num,
            data["r"].values,
            partition_mat_list,
            rep_mat_list,
            self.n_replicates,
            specular_rep_mat_list=specular_rep_mat_list,
        )
        return kernel

    def gp_pairing_fit(
        self,
        partition: List[List[str]],
        k_prior=False,
    ):
        """Function that fits the Pairing effect model on
        a specific partition of the condition set

        Args:
            partition (List[List[str]]): a partition of the condition set
            k_prior (bool, optional): to apply the hyper-prior distributions for the
                pairing effect model kernel parameters. Defaults to False.

        Returns:
            (GPy.models.GPRegression, float): (fitted pairing effect model,
                marginal likelihood)
        """
        data, _, _ = self.pairing_data(partition)
        kernel = self.generate_pairing_kernel(
            data,
            partition,
            k_prior,
        )

        X = data[["X", "r"]].values
        y = data[["y"]].values

        gp, score = self.gp_unit(X, y, kernel)
        return gp, score

    def get_conditions_data(
        self,
        subset: List[str],
        normalize: bool = False,
    ):
        """Filter data to return only the data related to a subset
        Also, normalization can be applied here on the data.

        Args:
            subset (List[str]): a subset of the a partition of the condition set
            normalize (bool, optional): to apply the normalization of the data
                Defaults to False.

        Returns:
            (pd.DataFrame, float, float): (data related to the specified partition
                of the condition set, mean of the data (if normalize==True else None),
                standard deviation of the data (if normalize==True else None))
        """
        conditions_idx = [self.conditions_dict[condition] for condition in subset]
        data = self.gene_data[self.gene_data["c"].isin(conditions_idx)]

        if self.timewarping:
            data["X"] = time_warping(data["X"])

        if normalize:
            mu, sigma = np.mean(data["y"]), np.std(data["y"])
            data["y"] = (data["y"] - mu) / sigma
        else:
            mu, sigma = None, None

        return data, mu, sigma

    def exact_prediction_full(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        time_pred: np.array,
        full_kern: GPy.kern.Kern,
        noise: float,
        partition: List[List[str]],
    ):
        """Given the data X, the gene expressions y and the fitted
        pairing effect kernel, this function applies the standard
        formula to calculate predictions (mean and standard deviation
        of the GP) on new time points.
            Rasmussen, Carl Edward. "Gaussian processes in machine learning."
            Summer School on Machine Learning. Springer, Berlin, Heidelberg, 2003.
            Chapter 2 - Equation (2.25),(2.26)
        The partition of the condition set
        needs to be specified. One can also specify noise==0 to get
        noiseless prediction.

        Args:
            X (pd.DataFrame): data frame containing (timepoints, replicate_num)
            y (pd.DataFrame): data frame with the corresponding gene expression
                values for each point in X
            time_pred (np.array): new time points for the predictions
            full_kern (GPy.kern.Kern): GPy kernel object to use for the predictions
            noise (float): noise value for the predictions formula
            partition (List[List[str]]): specify on which partition of the
                condition set the model was fit
        Returns:
            (np.array, np.array): mean and standard deviation of the precess on
                the new time points
        """
        Xnew, Xnew_partition_num, Xnew_replicate_num = self.generate_Xnew(
            time_pred, partition
        )
        K_XX = full_kern.K(X)

        if noise == 0:
            alpha = np.linalg.inv(K_XX + 1e-5 * np.eye(X.shape[0]))
        else:
            alpha = np.linalg.inv(K_XX + noise * np.eye(X.shape[0]))

        K_XnewX = full_kern.K(
            X,
            X2=Xnew,
            X2_partition_num=Xnew_partition_num,
            X2_rep_num=Xnew_replicate_num,
        )
        K_XnewXnew = full_kern.K(
            Xnew,
            X2=Xnew,
            X2_partition_num=Xnew_partition_num,
            X2_rep_num=Xnew_replicate_num,
            X_partition_num=Xnew_partition_num,
            X_rep_num=Xnew_replicate_num,
        )

        mean = K_XnewX.T.dot(alpha).dot(y)
        var = K_XnewXnew - K_XnewX.T.dot(alpha).dot(K_XnewX)

        return mean, np.diag(var)

    def exact_prediction_rep(self, X, y, time_pred, full_kern, noise, partition):
        """Given the data X, the gene expressions y and the fitted
        pairing effect kernel, this function applies the standard
        formula to calculate predictions (mean and standard deviation
        of the GP) on new time points. The difference with the function
        exact_prediction_full is thath the predictions for the response
        effect and for the pairing effect are returned separately.
        The partition of the condition set needs to be specified.
        One can also specify noise==0 to get noiseless prediction.

        Args:
            X (pd.DataFrame): data frame containing (timepoints, replicate_num)
            y (pd.DataFrame): data frame with the corresponding gene expression
                values for each point in X
            time_pred (np.array): new time points for the predictions
            full_kern (GPy.kern.Kern): GPy kernel object to use for the predictions
            noise (float): noise value for the predictions formula
            partition (List[List[str]]): specify on which partition of the
                condition set the model was fit
        Returns:
            (np.array, np.array, np.array, np.array): mean of the response effect,
                standard deviation of the response effect,
                mean of the pairing effect,
                standard deviation of the pairing effect
        """
        Xnew, Xnew_partition_num, Xnew_replicate_num = self.generate_Xnew(
            time_pred, partition
        )
        K_XX, Krep_XX = full_kern.compute_K_Krep(X)

        K_XnewX, Krep_XnewX = full_kern.compute_K_Krep(
            X,
            X2=Xnew,
            X2_partition_num=Xnew_partition_num,
            X2_rep_num=Xnew_replicate_num,
        )
        K_XnewXnew, Krep_XnewXnew = full_kern.compute_K_Krep(
            Xnew,
            X2=Xnew,
            X2_partition_num=Xnew_partition_num,
            X2_rep_num=Xnew_replicate_num,
            X_partition_num=Xnew_partition_num,
            X_rep_num=Xnew_replicate_num,
        )

        if noise == 0:
            alpha = np.linalg.inv(K_XX + Krep_XX + 1e-5 * np.eye(X.shape[0]))
        else:
            alpha = np.linalg.inv(K_XX + Krep_XX + noise * np.eye(X.shape[0]))

        mean_f = K_XnewX.T.dot(alpha).dot(y)
        var_f = K_XnewXnew - K_XnewX.T.dot(alpha).dot(K_XnewX)

        mean_r = Krep_XnewX.T.dot(alpha).dot(y)
        var_r = Krep_XnewXnew - Krep_XnewX.T.dot(alpha).dot(Krep_XnewX)

        return mean_f, np.diag(var_f), mean_r, np.diag(var_r)

    def exact_prediction_quantiles(
        self,
        var: np.array,
        quantiles: tuple = (2.5, 97.5),
    ):
        """Given the predictions for the variance of a GP,
        this function calculates the correspondent quantiles.

        Args:
            var (np.array): array containing the variance of the process
            quantiles (tuple): tuple with the quantiles that needs
                to be calculated

        Returns:
            List[np.array]: list with one numpy array for each calculated quantile

        """
        # return [stats.norm.ppf(q/100.)*np.sqrt(var + sigma2) for q in quantiles]
        return [stats.norm.ppf(q / 100.0) * np.sqrt(var) for q in quantiles]

    def generate_Xnew(self, time_pred: np.array, partition: List[List[str]]):
        """Considering the structure of the data, which include
        (timepoint, replicate_num, condition_num), and a specific
        partition of the condition set, this function returns
        3 arrays with the necessary data to perform predicitons:
            - one array with the time points
            - one array with the partition number for each time point
            - one array with the replicate number for each time point
        In other words, this function generate a set of time points for each
        replicate and for each partition which are concatenated in the resulting arrays.

        Args:
            time_pred (np.array): time points
            partition (List[List[str]]): partition of the condition set to be used
                to generate the correct structure

        Returns:
            (np.array, np.array, np.array): array with the time points,
                array with the partition numbers, array with the replicate numbers

        """
        Xnew = np.tile(time_pred, len(partition) * self.n_replicates)

        Xnew_partition_num = []
        Xnew_replicate_num = []
        for i, subset in enumerate(list(partition)):
            Xnew_partition_num.append(
                np.repeat(i + 1, len(time_pred) * self.n_replicates)
            )

            for r in range(self.n_replicates):
                Xnew_replicate_num.append(np.repeat(r + 1, len(time_pred)))
        Xnew_partition_num = np.concatenate(Xnew_partition_num)
        Xnew_replicate_num = np.concatenate(Xnew_replicate_num)

        Xnew = Xnew.reshape((-1, 1))
        Xnew_partition_num = Xnew_partition_num.reshape((-1, 1))
        Xnew_replicate_num = Xnew_replicate_num.reshape((-1, 1))

        Xnew = np.concatenate((Xnew, Xnew_replicate_num), axis=1)
        return Xnew, Xnew_partition_num, Xnew_replicate_num

    def full_predictions(
        self,
        data: pd.DataFrame,
        time_pred: np.array,
        gaussian_noise: float,
    ):
        """Wrapper for the function exact_prediction_full.
        Given the gene expression data, a set of time points
        and the level of gaussin noise, it calculates the predictions
        for the GP (mean, standard deviation, 2.5 quantile, 97.5 quantile)

        Args:
            data (pd.DataFrame): gene expression data
            time_pred (np.array): array with time points for predictions
            gaussian_noise (float): level of gaussin noise for the predictions

        Returns:
            (np.array,np.array,np.array,np.array): predictions on the specified time points
                (mean, standard deviation, 2.5 quantile, 97.5 quantile)

        """

        X = data[["X", "r"]].values
        y = data[["y"]].values
        mean, var = self.exact_prediction_full(
            X, y, time_pred, self.models.kern, gaussian_noise, self.partition
        )

        lower_interval, upper_interval = self.exact_prediction_quantiles(var)
        lower_interval, upper_interval = (
            lower_interval.flatten(),
            upper_interval.flatten(),
        )

        return mean, var, lower_interval, upper_interval

    def full_splitted_prediction(
        self,
        data: pd.DataFrame,
        time_pred: np.array,
        gaussian_noise: float,
    ):
        """Wrapper for the function exact_prediction_rep.
        Given the gene expression data, a set of time points
        and the level of gaussin noise, it calculates the predictions
        for the GP (mean, standard deviation, 2.5 quantile, 97.5 quantile).
        The results for the response effect and for the pairing effect
        are returned separately.

        Args:
            data (pd.DataFrame): gene expression data
            time_pred (np.array): array with time points for predictions
            gaussian_noise (float): level of gaussin noise for the predictions

        Returns:
            (np.array,np.array,np.array,np.array,np.array,np.array,np.array,np.array): predictions on the specified time points
                (mean, standard deviation, 2.5 quantile, 97.5 quantile) of the response effect
                (mean, standard deviation, 2.5 quantile, 97.5 quantile) of the pairing effect

        """

        X = data[["X", "r"]].values
        y = data[["y"]].values
        mean_f, var_f, mean_r, var_r = self.exact_prediction_rep(
            X, y, time_pred, self.models.kern, gaussian_noise, self.partition
        )

        lower_interval_f, upper_interval_f = self.exact_prediction_quantiles(var_f)
        lower_interval_f = lower_interval_f.flatten()
        upper_interval_f = upper_interval_f.flatten()

        lower_interval_r, upper_interval_r = self.exact_prediction_quantiles(var_r)
        lower_interval_r = lower_interval_r.flatten()
        upper_interval_r = upper_interval_r.flatten()

        return (
            mean_f,
            var_f,
            mean_r,
            var_r,
            lower_interval_f,
            upper_interval_f,
            lower_interval_r,
            upper_interval_r,
        )

    def normalize_gene_data(self, data):
        """Given a pd.DataFrame with a column 'y' containing
        the gene expression data, it returns the pd.DataFrame
        where the columns 'y' is normalized

        Args:
            data (pd.DataFrame): pd.DataFrame to normalize

        Returns:
            pd.DataFrame: normalized pd.DataFrame
        """
        mu, sigma = np.mean(data["y"]), np.std(data["y"])
        data["y"] = (data["y"] - mu) / sigma

        return data, mu, sigma

    def plot(
        self,
        noiseless: bool = False,
        title: str = None,
    ):
        """Plot the results of the pairing effect model

        Args:
            noiseless (bool, optional): to plot the predictions without including
                the Gaussian noise component of the GP model. Defaults to False:bool.
            title (str, optional): Alternative title for the plot. Defaults to None:str.
        """
        time_pred = self.get_time_pred()
        time_predN = time_pred.reshape((-1, 1))
        # print("Marginal log likelihood = " + str(self.models.log_likelihood()))

        if noiseless:
            gaussian_noise = 0
        else:
            gaussian_noise = self.models.Gaussian_noise[0]

        data, _, _ = self.get_conditions_data(self.conditions)

        ordered_data = pd.DataFrame([], columns=data.columns)
        ordered_conditions = []
        for subset in self.partition:
            for condition in subset:
                ordered_data = ordered_data.append(
                    data[data["c"] == self.conditions_dict[condition]]
                )
        data = ordered_data

        partition_num = self.generate_partition_number(self.partition)

        data["s"] = partition_num
        data, mu, sigma = self.normalize_gene_data(data)

        data = data.sort_values(["s", "r"])

        X = data[["X", "r"]]
        y = data[["y"]]

        mean, var, lower_interval, upper_interval = self.full_predictions(
            data, time_pred, gaussian_noise
        )

        # self.plot_base_plus_pairing(data, mu, sigma, time_pred, mean, lower_interval, upper_interval)
        # self.plot_base_plus_pairing_allr(data, mu, sigma, time_pred, mean, lower_interval, upper_interval)

        (
            mean_f,
            var_f,
            mean_r,
            _,
            low_f,
            up_f,
            low_r,
            up_r,
        ) = self.full_splitted_prediction(data, time_pred, gaussian_noise)

        self.plot_base(
            data, mu, sigma, time_pred, mean_f, var_f, low_f, up_f, title=title
        )
        # self.plot_base_separated(data, mu, sigma, time_pred, mean_f, var_f, lower_interval_f, upper_interval_f)

        self.plot_pairing_effect(
            data, mu, sigma, time_pred, mean_r, mean_f, low_r, up_r
        )
        # K, K_rep = self.plot_kernels(sigma, X.values)
        # return K, K_rep

    def plot_base(
        self,
        data: pd.DataFrame,
        mu: float,
        sigma: float,
        time_pred: np.array,
        mean_f: np.array,
        var_f: np.array,
        lower_interval_f: np.array,
        upper_interval_f: np.array,
        title: str = None,
    ):
        """Plot the resulting response effect (without the pairing effect)

        Args:
            data (pd.DataFrame): gene expression data
            mu (float): mean used to normalize the data before the model fit
            sigma (float): std used to normalize the data before the model fit
            time_pred (np.array): time points for the predictions
            mean_f (np.array): mean of the response kernels predictions
            var_f (np.array): variance of the response kernels predictions
            lower_interval_f (np.array): 5th percentile of the response
                kernels predictions
            upper_interval_f (np.array): 95th percentile of the response
                kernels predictions
            title (str, optional): Alternative title for the plot.
                Defaults to None:str.
        """
        plt.figure(figsize=(16, 7))
        replicate_idx = pd.unique(data["r"]).astype(int)
        for i in range(int(len(self.partition) * self.n_replicates)):
            if (i % self.n_replicates) == 0:
                a = int(i * len(time_pred))
                b = int((i + 1) * len(time_pred))

                mean_f_subset = mean_f[a:b].reshape(b - a)

                color_idx = i // self.n_replicates
                c = self.colors[color_idx]

                plt.plot(
                    time_pred,
                    (mean_f_subset * sigma) + mu,
                    color=c,
                    label=get_label(self.partition[color_idx]),
                )

                lower = mean_f_subset + lower_interval_f[a:b]
                upper = mean_f_subset + upper_interval_f[a:b]

                lower = (lower * sigma) + mu
                upper = (upper * sigma) + mu

                plt.plot(time_pred, lower, color=c, linestyle="--")
                plt.plot(time_pred, upper, color=c, linestyle="--")
                plt.fill_between(time_pred, lower, upper, color=c, alpha=0.05)

        for i, subset in enumerate(self.partition):
            subset_data = data[data["s"] == (i + 1)]  # self.get_conditions_data(subset)
            for r_idx, r in enumerate(replicate_idx):
                r_data = subset_data[subset_data["r"] == r]  # (r + 1)]
                data_points = (r_data["y"].values * sigma) + mu
                plt.scatter(
                    r_data["X"].values,
                    data_points,
                    marker=self.markers[r_idx],
                    c=self.colors[i],
                    # label="replicate " + str(r),
                )

        if title is None:
            plt.title("" + self.gene_name + "")
        else:
            plt.title(title)
        if self.timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel("gene expression level", fontsize=22)
        plt.legend(prop={"size": 24})
        plt.grid()
        # plt.ylim([1.3,2.8])
        # plt.savefig("gene_figures/" + self.gene_name + "_pair_model.svg")
        # plt.tight_layout()
        plt.tight_layout(rect=[0.015, 0, 1, 1])
        # plt.savefig("gene_figures/" + self.gene_name + "_pair_model.jpg", format="jpg")
        plt.show()

    def plot_pairing_effect(
        self,
        data: pd.DataFrame,
        mu: float,
        sigma: float,
        time_pred: np.array,
        mean_r: np.array,
        mean_f: np.array,
        lower_interval_r: np.array,
        upper_interval_r: np.array,
    ):
        """Plot the resulting pairing effect for each replicate

        Args:
            data (pd.DataFrame): gene expression data
            mu (float): mean used to normalize the data before the model fit
            sigma (float): std used to normalize the data before the model fit
            time_pred (np.array): time points for the predictions
            mean_f (np.array): mean of the response kernels predictions
            var_f (np.array): mean of the pairing effect kernels predictions
            lower_interval_r (np.array): 5th percentile of the pairing effect
                kernels predictions
            upper_interval_r (np.array): 95th percentile of the pairing effect
                kernels predictions
        """
        plt.figure(figsize=(16, 7))

        # mean = ((mean_f+mean_r)*sigma)+mu
        # mean_f_ = (mean_f*sigma)+mu
        # mean_r = mean - mean_f_
        mean_r_rescaled = mean_r * sigma

        for i in range(int(self.n_replicates)):
            a, b = int(i * len(time_pred)), int((i + 1) * len(time_pred))

            mean_r_subset = mean_r[a:b].reshape(b - a)

            color_idx = i % self.n_replicates
            c = self.repcolors[color_idx]

            mean_r_subset_denormalized = mean_r_subset * sigma
            plt.plot(time_pred, mean_r_subset_denormalized, color=c)
            plt.scatter(
                time_pred[::10],
                mean_r_subset_denormalized[::10],
                color=c,
                marker=self.markers[i],
                label="replicate " + str(color_idx + 1),
            )

            lower = mean_r_subset + lower_interval_r[a:b]
            upper = mean_r_subset + upper_interval_r[a:b]

            plt.plot(time_pred, lower * sigma, color=c, linestyle="--")
            plt.plot(time_pred, upper * sigma, color=c, linestyle="--")
            plt.fill_between(
                time_pred, lower * sigma, upper * sigma, color=c, alpha=0.05
            )

        plt.title("Pairing effect")
        # plt.legend(prop={'size': 24})
        plt.legend(
            prop={"size": 24},
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        """
        plt.legend(
            prop={'size': 24},
            bbox_to_anchor=(0.02, 0.98),
            loc='upper left',
            ncol=3,
            borderaxespad=0.
        )
        """
        if self.timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.grid()
        # plt.yticks([-0.5,0,0.5,1])
        plt.tight_layout()
        # plt.savefig("gene_figures/" + self.gene_name + "_pair_eff.jpg", format="jpg")
        plt.show()

    def plot_kernels(self, sigma: float, X=None):
        """Utility to plot the kernels of the pairing effect model.
        The plot diplays the response kernel, the pairing effect kernel
        and their sum.

        Args:
            sigma (float): standard deviation used to standardize the data
            X (pd.DataFrame): dataset containing (time point, replicate number)
                for each data point
        """
        if np.any(X != None):
            X = X
        else:
            data, _, _ = self.get_conditions_data(self.conditions)
            partition_num = self.generate_partition_number(self.partition)

            data["s"] = partition_num
            data = data.sort_values(["s", "r"])
            X = data[["X", "r"]].values

        plt.figure(figsize=(16, 5))
        plt.subplot(131)
        plt.imshow(self.models.kern.K(X) * (sigma ** 2), cmap="hot")
        plt.title("$K_r+K_p$")

        plt.colorbar()

        K, K_rep = self.models.kern.compute_K_Krep(X)
        plt.subplot(132)
        plt.imshow(K * (sigma ** 2), cmap="hot")
        plt.title("$K_r$")

        plt.colorbar()

        plt.subplot(133)
        plt.imshow(K_rep * (sigma ** 2), cmap="hot")

        plt.colorbar()
        plt.title("$K_p$")
        plt.tight_layout()
        # plt.savefig("gene_figures/" + self.gene_name + "_k.jpg", format="jpg")
        plt.show()

        return K, K_rep
