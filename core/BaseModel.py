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

from DiffConditionModel import DiffConditionModel

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

class BaseModel(DiffConditionModel):
    def __init__(
        self,
        *args, 
        **kwargs
    ):
        super(BaseModel, self).__init__(*args, **kwargs)
        # we define here the hyperprior distributions for the model.
        # Refer to GPy documentation https://gpy.readthedocs.io/en/deploy/GPy.kern.src.html
        # for the underlying formulas.
        # Prior distribution for the lengthscale parameter of the kernel
        self.prior_distribution_f_ls = GPy.priors.LogGaussian(0.5,0.5)
        
        # these attributes keep the state of the object
        # by referring to the best model obtained through
        # the model selection precess (the fit function)
        self.mll = None
        self.models = None
        self.partition = None

    def fit(
        self,
        timewarping: bool=True, 
        k_prior: bool=True,
        use_stored: bool=False,
        use_gpu: bool=False,
        verbose: bool=True,
    ):
        """Model selection process for the base model
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
            use_stored (bool, optional): If the model selection process has already
                been performed, then one can just load the results. Defaults to False:bool.
            use_gpu (bool, optional): The Gaussian kernel used in this application
                allow to perform the computations also on GPU. Defaults to False:bool.
            verbose (bool, optional): Used to log more information during the model
                selection process in the terminal. Defaults to True:bool.
        """
        self.timewarping = timewarping
        self.use_stored = use_stored
        self.use_gpu = use_gpu

        if use_stored == False:
            # remove all the already trained model
            self.delete_models()

        partition_models = []
        for p in tqdm(self.partitions, disable=(not verbose)):
            mll_p = []
            models = []
            for subset in p:
                name = self.get_model_file_name(subset)
                if os.path.isfile(name):
                    gp, score = load_gp_model(name)
                else:
                    # train a GP on that set of conditions
                    kernel = self.generate_base_kernel()
                    gp, score = self.gp_base_fit(kernel, subset, k_prior=k_prior)
                    save_gp_model((gp, score), name)
                mll_p.append(score)
                models.append(gp)
            mll = np.sum(mll_p)
            partition_models.append([p, mll])

            if self.mll == None or mll > self.mll:
                self.mll = mll
                self.models = models
                self.partition = p
        self.partitions_ranking = pd.DataFrame(
            partition_models,
            columns=["p", "score"]
        ).sort_values("score", ascending=False)
    

    def gp_base_fit(
        self,
        kernel,
        subset,
        k_prior=False
    ):
        data, _, _ = self.get_conditions_data(subset)#, normalize=True)

        X = data[["X"]]
        y = data[["y"]]

        if k_prior:
            kernel.lengthscale.set_prior(
                self.prior_distribution_f_ls,
                warning=False
            )

        gp, score = self.gp_unit(
            X,
            y,
            kernel,
            normalizer=True
        )

        return gp, score

    
    def generate_base_kernel(self):
        kernel = GPy.kern.RBF(
            input_dim=1,
            useGPU=self.use_gpu,
            name="f"
        )

        return kernel


    def plot(
        self,
        title=None
    ):
        time_pred = self.get_time_pred()
        time_predN = time_pred.reshape((-1, 1))

        replicate_idx = pd.unique(self.gene_data['r']).astype(int)

        plt.figure(figsize=(16,7))
        for i, subset in enumerate(self.partition):
            label = get_label(subset)

            y_pred_mean_f, y_pred_var_f = self.models[i].predict(time_predN)
            lower_interval, upper_interval = self.models[i].predict_quantiles(time_predN)

            y_pred_mean_f = y_pred_mean_f.reshape(len(y_pred_mean_f))

            upper_interval = upper_interval.flatten()
            lower_interval = lower_interval.flatten()

            #upper_interval = np.exp(upper_interval)-1
            #lower_interval = np.exp(lower_interval)-1

            plt.plot(
                time_pred,
                upper_interval,
                color=self.colors[i],
                linestyle="--"
            )
            plt.plot(
                time_pred,
                lower_interval,
                color=self.colors[i],
                linestyle="--"
            )
            plt.fill_between(
                time_pred,
                lower_interval,
                upper_interval,
                color=self.colors[i],
                alpha=0.05
            )

            variance = self.models[i].kern.to_dict()['variance'][0]
            lengthscale = self.models[i].kern.to_dict()['lengthscale'][0]
            #label += '\n(alpha=%.2f , l=%.2f)' % (variance, lengthscale)

            #y_pred_mean_f = np.exp(y_pred_mean_f)-1
            #print("about to plot")
            #print(time_pred)
            #print(y_pred_mean_f)
            plt.plot(
                time_pred,
                y_pred_mean_f,
                self.colors[i],
                label=label
            )

            data, _, _ = self.get_conditions_data(subset)  # , normalize=True)
            for r_idx, r in enumerate(replicate_idx):
                r_data = data[data["r"] == r]
                plt.scatter(
                    r_data["X"].values,
                    r_data["y"].values,
                    marker=self.markers[r_idx],
                    c=self.colors[i],
                    #label="replicate " + str(r),
                )

        if self.timewarping:
            plt.xlabel("log Time", fontsize=22)
        else:
            plt.xlabel("Time")
        plt.ylabel(
            "gene expression level (log(y+1))",
            fontsize=22
        )  # [log(y+1)]")

        mll = "Marginal likelihood = %.2f" % self.mll

        if title is None:
            plt.title("" + self.gene_name + "")
        else:
            plt.title(title)
        plt.legend(prop={'size': 24})
        #plt.legend(
        # prop={'size': 24},
        # bbox_to_anchor=(1.02, 1),
        # loc='upper left',
        # borderaxespad=0.
        # )
        plt.grid()
        plt.tight_layout()
        #plt.savefig(
        #    "gene_figures/" + self.gene_name + "_base_model.jpg",
        #    format="jpg"
        #)

        plt.show()


    def get_model_file_name(
        self,
        subset,
        pairing=False
    ):
        name = self.models_folder + "/" + self.gene_name + "/"

        if not os.path.exists(name):
            os.makedirs(name)

        for condition in subset:
            name += condition
        if pairing:
            name += "_pairing"
        name += ".pkl"
        return name

