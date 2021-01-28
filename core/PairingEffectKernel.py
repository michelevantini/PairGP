import GPy
import numpy as np
from typing import List


def get_partition_mat(
    X_partition_num: np.array,
    X2_partition_num: np.array,
    get_specular: bool=False,
):
    """Assuming to have two sets of samples organized in the data 
    as in X_partition_num and X2_partition_num, then this function 
    returns two lists of matrices. The lists contain one matrix for each
    partition number (numpy.unique(partition_num)):
        1) a matrix with ones where the partition numbers match 
        2) a matrix with ones where the partition numbers do not match,
          but only looking at the the rows where they match

    e.g., partitio_num = [1,2,3,1,2,3]
    we assume to put this list matching with the rows and columns 
    of a matrix:
          [1,1,2,2,3,3]
       [1,
        2,
        3,
        1,
        2,
        3]
    then the first elements for the two lists are the following
    1) where they match:
        [[1 1 0 0 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]
         [1 1 0 0 0 0]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
    2) where they don't, but looking only at the rows where they match:
        [[0 0 1 1 1 1]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]
         [0 0 1 1 1 1]
         [0 0 0 0 0 0]
         [0 0 0 0 0 0]]
    
    This is a utility for the centered kernel calculations.
          

    Args:
        X_partition_num (np.array): array containing the partition number
            for each sample in a gene expression dataset
        X2_partition_num (np.array): array containing the partition number
            for each sample in a gene expression dataset
        get_specular (bool, optional): to get both the the matrices lists (True)
        or only the list number 1), as explained above (False) . Defaults to False.

    Returns:
        (np.array, np.array): the resulting matrices lists
    """
    subset_mat_list = []
    specular_mat_list = []
    subset_idx = np.unique(X_partition_num)
    for idx in subset_idx:
        X_subset_idx_arr = (X_partition_num * (X_partition_num == idx)) / idx
        X2_subset_idx_arr = (X2_partition_num * (X2_partition_num == idx)) / idx
        res = np.outer(
            X_subset_idx_arr,
            X2_subset_idx_arr
        )
        subset_mat_list.append(res)

        X_partition_num_idx = (X_partition_num == idx).astype(bool)
        X2_partition_num_idx = (X2_partition_num != idx).astype(bool)
        res = np.outer(
            X_partition_num_idx,
            X2_partition_num_idx
        )
        specular_mat_list.append(res)

    if get_specular:
        return subset_mat_list, specular_mat_list
    else:
        return subset_mat_list


class PairingEffectKernel(GPy.kern.src.kern.CombinationKernel):

    def __init__(
        self,
        kernels: List[GPy.kern.Kern],
        partition_num: np.array,
        rep_num: np.array,
        partition_mat_list: List[np.array],
        replicate_mat_list: List[np.array],
        n_replicates: int,
        centered_kernel: bool=True,
        specular_rep_mat_list: List[np.array]=None,
        name: str='hierarchy',
    ):
        """Pairing effect kernel constructor

        Args:
            kernels (List[GPy.kern.Kern]): list of kernels, one for the response
                effect modeling and one for the pairing effect modeling
            partition_num (np.array): array containing the partition number
                for each sample in the gene expression dataset
            rep_num (np.array): array containing the replicate number
                for each sample in the gene expression dataset
            partition_mat_list (List[np.array]): first list of matrices obtained through
                calling get_partition_mat(partition_num)
            replicate_mat_list (List[np.array]): first list of matrices obtained through
                calling get_partition_mat(rep_num)
            n_replicates (int): total number of replicates in the experiment
            centered_kernel (bool, optional): The default behaviour is to calculate
                the pairing effect kernel so that the pairing effect is centered
                around the response effect. Defaults to True.
            specular_rep_mat_list (List[np.array], optional): second list of matrices 
                obtained through calling get_partition_mat(rep_num). Defaults to None.
            name (str, optional): Name of the kernel. Defaults to 'hierarchy'.
        """

        kernels_flatten = kernels[0] + [kernels[1]]
        assert all([k.input_dim==kernels_flatten[0].input_dim for k in kernels_flatten])
        assert len(kernels) > 1
        self.f = kernels[0]
        self.g = kernels[1]
        self.levels = len(kernels) -1
        self.subset_mat_list = partition_mat_list
        self.replicate_mat_list = replicate_mat_list

        self.centerd_kernel=centered_kernel
        self.specular_rep_mat_list = specular_rep_mat_list

        self.partition_num = partition_num
        self.rep_num = rep_num
        self.n_replicates = n_replicates

        self.eval_count = 0

        input_max = max([k.input_dim for k in kernels_flatten])
        super(PairingEffectKernel, self).__init__(
            kernels=kernels_flatten,
            extra_dims = range(input_max, input_max + len(kernels)-1),
            name=name
        )

    def K(
        self,
        X: np.array,
        X2: np.array=None,
        X_partition_num: np.array=None,
        X_rep_num: np.array=None,
        X2_partition_num: np.array=None,
        X2_rep_num: np.array=None,
    ):
        """Calculate the kernel values K(X,X) (when X2=None) or K(X,X2)

        Args:
            X (np.array): data frame containing (timepoints, replicate_num) for
                a gene expression dataset
            X2 (np.array, optional): data frame containing (timepoints, replicate_num) for
                a gene expression dataset. Defaults to None.
            X_partition_num (np.array, optional): array containing the partition number
                for each sample in the gene expression dataset in X. Defaults to None.
            X_rep_num (np.array, optional): array containing the replicate number
                for each sample in the gene expression dataset in X. Defaults to None.
            X2_partition_num (np.array, optional): array containing the partition number
                for each sample in the gene expression dataset in X2. Defaults to None.
            X2_rep_num (np.array, optional): array containing the replicate number
                for each sample in the gene expression dataset in X. Defaults to None.

        Returns:
            np.array: Kernel obtained by the sum of the response effect
                kernel and the pairing effect kernel
        """
        K, K_rep = self.compute_K_Krep(
            X,
            X2,
            X_partition_num,
            X_rep_num,
            X2_partition_num,
            X2_rep_num
        )

        return K + K_rep

    def compute_K_Krep(
        self,
        X,
        X2=None,
        X_partition_num=None,
        X_rep_num=None,
        X2_partition_num=None,
        X2_rep_num=None
    ):
        """Calculate the kernel values K(X,X) (when X2=None) or K(X,X2)
        but returning the response effect kernel and the pairinge effect
        kernel as separate components.

        Args:
            X (np.array): data frame containing (timepoints, replicate_num) for
                a gene expression dataset
            X2 (np.array, optional): data frame containing (timepoints, replicate_num) for
                a gene expression dataset. Defaults to None.
            X_partition_num (np.array, optional): array containing the partition number
                for each sample in the gene expression dataset in X. Defaults to None.
            X_rep_num (np.array, optional): array containing the replicate number
                for each sample in the gene expression dataset in X. Defaults to None.
            X2_partition_num (np.array, optional): array containing the partition number
                for each sample in the gene expression dataset in X2. Defaults to None.
            X2_rep_num (np.array, optional): array containing the replicate number
                for each sample in the gene expression dataset in X. Defaults to None.

        Returns:
            (np.array,np.array): the response effect kernel and the pairing effect kernel
        """
        X = X[:, [0]]

        if np.any(X2 != None):
            X2 = X2[:, [0]]

            if np.any(X_partition_num != None):
                subset_mat_list = get_partition_mat(
                    X_partition_num,
                    X2_partition_num
                )
                replicate_mat_list, specular_mat_list = get_partition_mat(
                    X_rep_num,
                    X2_rep_num,
                    get_specular=True
                )
            else:
                subset_mat_list = get_partition_mat(
                    self.partition_num,
                    X2_partition_num
                )
                replicate_mat_list, specular_mat_list = get_partition_mat(
                    self.rep_num,
                    X2_rep_num,
                    get_specular=True
                )
        else:
            X2 = X[:, [0]]

            subset_mat_list = self.subset_mat_list
            replicate_mat_list = self.replicate_mat_list
            specular_mat_list = self.specular_rep_mat_list

        # BASE KERNEL
        K = np.zeros((X.shape[0], X2.shape[0]))

        for i in range(len(self.subset_mat_list)):
            subset_K = self.f[i].K(X, X2) * subset_mat_list[i]

            K += subset_K

        # PAIRING KERNEL
        K_rep = np.zeros((X.shape[0], X2.shape[0]))

        for i in range(len(self.replicate_mat_list)):
            subset_K_rep = self.g.K(X, X2) * replicate_mat_list[i] #+ 1e-6
            if self.centerd_kernel:
                subset_K_rep += (-1/self.n_replicates) * self.g.K(X, X2) * specular_mat_list[i]

            K_rep += subset_K_rep

        return K, K_rep

    def Kdiag(
        self,
        X
    ):
        return np.diag(self.K(X))

    def gradients_X(
        self,
        dL_dK,
        X,
        X2=None
    ):
        raise NotImplementedError

    def get_matrix_subset(
        self,
        dL_dK: np.array,
        bool_mask: np.array,
    ):
        """Get the subelements indicated in bool_mask from
        dL_dK and reorganize them in a square matrix.

        Args:
            dL_dK (np.array): a matrix
            bool_mask (np.array): a boolean mask

        Returns:
            np.array: the resulting square matrix
        """
        submat = dL_dK[bool_mask.astype(bool)]
        n = int(np.sqrt(len(submat)))
        res = submat.reshape(n,n)
        return res

    def get_X_subset(
        self,
        X: np.array,
        bool_mask: np.array,
    ):
        """Get the subelements indicated 
        in bool_mask from X.

        Args:
            X (np.array): a matrix
            bool_mask (np.array): a boolean mask

        Returns:
            np.array: the resulting matrix
        """
        idx = np.arange(0, X.shape[0], 1)
        subset_idx = idx[bool_mask].astype(int)
        return X[subset_idx]

    def update_gradients_full(
        self,
        dL_dK: np.array,
        X: np.array,
        X2: np.array=None
    ):
        """[summary]

        Args:
            dL_dK (np.array): matrix with the current derivatives
            of the marginal likelihood with the respect to the kernel
            X (np.array): data on which the optimization is run
            X2 (np.array, optional): X2!=None doesn't need to be supported
                for the purpose of this software. Defaults to None.

        Raises:
            NotImplementedError: X2!=None doesn't need to be supported
                for the purpose of this software
        """
        if X2 is None:
            X = X[:, [0]]

            for i in range(len(self.subset_mat_list)):
                dK_subset = self.get_matrix_subset(
                    dL_dK,
                    self.subset_mat_list[i]
                )
                X_subset = self.get_X_subset(
                    X,
                    self.partition_num == (i+1)
                )
                
                self.f[i].update_gradients_full(
                    dK_subset,
                    X_subset,
                    None
                )

            if self.centerd_kernel:
                dK_subset = dL_dK
            else:
                dK_subset = dL_dK * self.replicate_mat_list[0]
                for j in range(1, len(self.replicate_mat_list)):
                    dK_subset += dL_dK * self.replicate_mat_list[j]
                    #dK_subset = self.get_matrix_subset(dL_dK, self.replicate_mat_list[j])
                    #X_subset = self.get_X_subset(X, self.rep_num == (j+1))

            self.g.update_gradients_full(
                dK_subset,
                X,
                None
            )
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

        name = "g"
        var_name = name + ".variance"
        res[var_name] = self.g.variance[0]
        ls_name = name + ".lengthscale"
        res[ls_name] = self.g.lengthscale[0]
        return res