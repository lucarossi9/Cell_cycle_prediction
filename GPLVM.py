from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np
from torch.nn import Parameter
import gpytorch
import math
from typing import Optional
import cyclum.models
import cyclum.evaluation
from gpytorch.models import ApproximateGP
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.priors import NormalPrior
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from gpytorch.functions.matern_covariance import MaternCovariance
from gpytorch.kernels import Kernel
from gpytorch.means import ZeroMean
from gpytorch.settings import trace_mode
from tqdm.notebook import trange
from sklearn import preprocessing
import numba
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap
from mpl_toolkits.mplot3d import Axes3D

# MODEL
class ModifiedKernel(Kernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[float] = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(ModifiedKernel, self).__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1 = torch.cat((x1, 
                            torch.sin(x1), 
                            torch.cos(x1)), 1)
            x2 = torch.cat((x2, 
                            torch.sin(x2), 
                            torch.cos(x2)), 1)
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )


# More Scalable version
class bGPLVM(ApproximateGP):
    def __init__(self, n_inducing_points, n_latent_dims, n_data_points, n_data_dims, X_prior_mean):
        batch_shape = torch.Size([n_data_dims])
        inducing_points = torch.randn(n_data_dims, n_inducing_points, n_latent_dims)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape,
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super(bGPLVM, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            gpytorch.kernels.PeriodicKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )
        # self.covar_module = ModifiedKernel(nu=0.5, batch_shape=batch_shape)
        # self.covar_module = ScaleKernel(
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)),
        #     batch_shape=batch_shape
        # )
        self.X = Parameter(X_prior_mean.clone())
        self.register_parameter(
            name="X", 
            parameter=self.X
            )
        self.register_prior('prior_X', NormalPrior(X_prior_mean,torch.ones_like(X_prior_mean)), 'X')
        
    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

### Robust version
# class bGPLVM(BayesianGPLVM):
#     def __init__(self, n_data_points, data_dim, latent_dim, n_inducing, X_init):
#         self.n_data_points = n_data_points
#         self.batch_shape = torch.Size([data_dim])

#         # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
#         # regularly placed with shape (D x n_inducing x latent_dim).
#         self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

#         # Sparse Variational Formulation (inducing variables initialised as randn)
#         q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
#         q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

#         # Define prior for X
#         X_prior_mean = torch.zeros(n_data_points, latent_dim)  # shape: N x Q
#         prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))
#         # LatentVariable (c)
#         X = gpytorch.models.gplvm.VariationalLatentVariable(n_data_points, data_dim, latent_dim, X_init, prior_x)
#         super().__init__(X, q_f)
#         # Kernel (acting on latent dimensions)
#         self.mean_module = ZeroMean(ard_num_dims=latent_dim)
#         self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

#     def forward(self, X):
#         mean_x = self.mean_module(X)
#         covar_x = self.covar_module(X)
#         dist = MultivariateNormal(mean_x, covar_x)
#         return dist

#     def _get_batch_idx(self, batch_size):
#         valid_indices = np.arange(self.n_data_points)
#         batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
#         return np.sort(batch_indices)
    

# DATA LOADING
def load_data(dataset, filepath):
    if dataset == "H9":
        cell_line = "H9"
        raw_Y = pd.read_pickle(filepath+'/h9_df.pkl').T
        cpt = pd.read_pickle(filepath+'/h9_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "mb":
        cell_line = "mb"
        raw_Y = pd.read_pickle(filepath+'/mb_df.pkl').T
        cpt = pd.read_pickle(filepath+'/mb_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    elif dataset == "pc3":
        cell_line = "pc3"
        raw_Y = pd.read_pickle(filepath+'/pc3_df.pkl').T
        cpt = pd.read_pickle(filepath+'/pc3_cpt.pkl').values
        print("DATASET: ", cell_line)
        print("Original dimesion %d cells x %d genes." % raw_Y.shape)
        print(f"G0/G1 {sum(cpt == 'g0/g1')}, S {sum(cpt == 's')}, G2/M {sum(cpt == 'g2/m')}")
    else:
        raise NotImplementedError("Unknown dataset {dataset}")
    
    return raw_Y, cpt

# selecting the dat & setting the  informative prior
dataset = "pc3" 
filepath = '/Statistical_computation/project4/data/McDavid'
data, cpt = load_data(dataset, filepath)
data = data.to_numpy()
data = preprocessing.scale(data)
data = torch.tensor(data, dtype=torch.get_default_dtype())
df = pd.read_csv(filepath+'/GPrix_prior-pc3.csv')
x_mean = df['prior'].to_numpy()
x_mean = torch.tensor(x_mean, dtype=torch.get_default_dtype())


# Alternative prior with UMAP & circular metric
@numba.njit()
def circular(x, y):
    
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    result = np.sqrt(sin_lat ** 2 + np.cos(x[0]) * np.cos(y[0]))
    return 2.0 * np.arcsin(result)

@numba.njit()
def circular_grad(x, y):
    
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    cos_lat = np.cos(0.5 * (x[0] - y[0]))
    a_0 = np.cos(x[0] + np.pi / 2) * np.cos(y[0] + np.pi / 2)
    a_1 = a_0 + sin_lat ** 2
    d = 2.0 * np.arcsin(np.sqrt(min(max(abs(a_1), 0), 1)))
    denom = np.sqrt(abs(a_1 - 1)) * np.sqrt(abs(a_1))
    grad = (
        np.array(
            [
                (
                    sin_lat * cos_lat
                    - np.sin(x[0] + np.pi / 2)
                    * np.cos(y[0] + np.pi / 2)
                    
                ),
                (
                    np.cos(x[0] + np.pi / 2)
                    * np.cos(y[0] + np.pi / 2)
                ),
            ]
        )
        / (denom + 1e-6)
    )
    return d, grad

reducer = umap.UMAP(
    n_components=1, n_neighbors=3, output_metric=circular_grad, transform_seed=42, verbose=False
)
reducer.fit(data, x_mean)
x = reducer.transform(data)
x_mean = (x % (2 * np.pi)) / 2
x_mean = torch.tensor(x_mean, dtype=torch.get_default_dtype())


# training
n_latent_dims = 1
model = bGPLVM(n_inducing_points=32, 
                n_latent_dims=n_latent_dims, 
                n_data_points = data.shape[0], 
                n_data_dims = data.shape[1],
                X_prior_mean = x_mean)
# model = bGPLVM(n_data_points = data.shape[0], 
#                 data_dim = data.shape[1], 
#                 latent_dim = n_latent_dims, 
#                 n_inducing = 32, 
#                 X_init = x_mean)


likelihood = GaussianLikelihood(num_tasks=data.shape[1], batch_shape=torch.Size([data.shape[1]]))
model.train()
likelihood.train()
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

mll = PredictiveLogLikelihood(likelihood, model, num_data=data.size(0))

loss_list = []
iterator = trange(300)
for i in iterator:
    optimizer.zero_grad()
    output = model(model.X)
    inp = model.X
    loss = -mll(output, data.T).sum() #-compute_accuracy(inp)
    loss_list.append(loss.item())
    print(str(loss.item()) + ", iter no: " + str(i))
    iterator.set_postfix(loss=loss.item())
    loss.backward(retain_graph=True)
    optimizer.step()
        
pseudotime = model.X.detach().numpy()
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score: ", dis_score)

#Writing to csv
cpt = cpt
cpt = np.concatenate(cpt).astype(str)
pseudotime = (pseudotime % (2 * np.pi))
data2 = np.matrix([pseudotime[:,0], cpt])
data2 = data2.T
df = pd.DataFrame(data=data2, columns=["time", "phase"])
df.to_csv(r'/home/pau/Desktop/MASTER/Statistical_computation/project4/visual-pc3.csv', index = False)

