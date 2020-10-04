import GPy
import numpy as np


def get_partition_mat(X_partition_num, X2_partition_num, get_specular=False):
    subset_mat_list = []
    specular_mat_list = []
    subset_idx = np.unique(X_partition_num)
    for idx in subset_idx:
        X_subset_idx_arr = (X_partition_num * (X_partition_num == idx)) / idx
        X2_subset_idx_arr = (X2_partition_num * (X2_partition_num == idx)) / idx
        res = np.outer(X_subset_idx_arr, X2_subset_idx_arr)
        subset_mat_list.append(res)

        #plot_kernel(res)

        X_partition_num_idx = (X_partition_num == idx).astype(bool)
        X2_partition_num_idx = (X2_partition_num != idx).astype(bool)
        res = np.outer(X_partition_num_idx, X2_partition_num_idx)
        specular_mat_list.append(res)

        #plot_kernel(res)

    if get_specular:
        return subset_mat_list, specular_mat_list
    else:
        return subset_mat_list

def plot_kernel(k):
    import matplotlib.pyplot as plt
    plt.imshow(k)
    plt.colorbar()
    plt.show()

class BaseKernel(GPy.kern.src.kern.CombinationKernel):

    def __init__(self, kernels, subset_num, partition_mat_list, name='hierarchy'):
        kernels_flatten = kernels
        assert all([k.input_dim==kernels_flatten[0].input_dim for k in kernels_flatten])
        #assert len(kernels) > 1
        self.f = kernels

        self.subset_mat_list = partition_mat_list
        self.subset_num = subset_num

        self.eval_count = 0

        input_max = max([k.input_dim for k in kernels_flatten])
        super(BaseKernel, self).__init__(kernels=kernels_flatten
                                                  , extra_dims = range(input_max, input_max)
                                                  , name=name)

    def K(self, X, X2=None, X_subset_num=None, X2_subset_num=None):
        K = self.compute_K_Krep(X, X2, X_subset_num, X2_subset_num)

        return K

    def compute_K_Krep(self, X, X2=None, X_subset_num=None, X2_subset_num=None):
        X = X[:, [0]]

        if np.any(X2 != None):
            X2 = X2[:, [0]]

            if np.any(X_subset_num != None):
                subset_mat_list = get_partition_mat(X_subset_num, X2_subset_num)
            else:
                subset_mat_list = get_partition_mat(self.subset_num, X2_subset_num)
        else:
            X2 = X[:, [0]]

            subset_mat_list = self.subset_mat_list

        # BASE KERNEL
        K = np.zeros((X.shape[0], X2.shape[0]))

        for i in range(len(self.subset_mat_list)):
            subset_K = self.f[i].K(X, X2) * subset_mat_list[i]

            K += subset_K

        return K

    def Kdiag(self,X):
        return np.diag(self.K(X))

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def get_matrix_subset(self, dL_dK, bool_mask):
        submat = dL_dK[bool_mask.astype(bool)]
        n = int(np.sqrt(len(submat)))
        res = submat.reshape(n,n)
        return res

    def get_X_subset(self, X, bool_mask):
        idx = np.arange(0, X.shape[0], 1)
        subset_idx = idx[bool_mask].astype(int)
        return X[subset_idx]

    def update_gradients_full(self,dL_dK,X,X2=None):
        if X2 is None:
            X = X[:, [0]]

            for i in range(len(self.subset_mat_list)):
                dK_subset = self.get_matrix_subset(dL_dK, self.subset_mat_list[i])
                X_subset = self.get_X_subset(X, self.subset_num == (i+1))
                #plot_kernel(dK_subset)
                #plot_kernel(X_subset)
                self.f[i].update_gradients_full(dK_subset, X_subset, None)
        else:
            raise NotImplementedError

    def to_dict(self):
        res = {}
        for i, f_kern in enumerate(self.f):
            name = "f" + str(i)
            var_name = name+".variance"
            res[var_name] = f_kern.variance[0]
            ls_name = name + ".lengthscale"
            res[ls_name] = f_kern.lengthscale[0]
        return res