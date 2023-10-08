#%%
# Credits To : https://towardsdatascience.com/gaussian-mixture-model-clearly-explained-115010f7d4cf
# Rewritten by utilizing static typing :3 & and cleaning code 
# .ipynb is copied from the this .py 

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.stats import multivariate_normal, norm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

#%%
DATA_1_MEAN : int = 0
DATA_1_STD : int = 1
DATA_1_SIZE : int = 1000

DATA_2_MEAN : int = 5
DATA_2_STD : int = 1
DATA_2_SIZE : int = 1000

BINS : int = 100

data1 : np.array = np.random.normal(DATA_1_MEAN, DATA_1_STD, DATA_1_SIZE)
data2 : np.array = np.random.normal(DATA_2_MEAN, DATA_2_STD, DATA_2_SIZE)

## Assummed the dataset is normally distributed
sns.distplot(data1, kde= True, hist=True, bins=BINS, color='b', hist_kws={'alpha' : 0.6})
sns.distplot(data2, kde= True, hist=True, bins=BINS, color='r', hist_kws={'alpha' : 0.6})
plt.title('Multi Modal Distributions')
plt.legend(['Rand Normal Data 1', 'Rand Normal Data 2'])
plt.show()

#%%

DATA_1_MULTIVARIATE_MEAN : List[int] = [0, 0 ]
DATA_2_MULTIVARIATE_MEAN : List[int] = [2, 0]

DATA_1_COVARIANCE : List[List[int]] = [[1, .7], [.7, 1]]
DATA_2_COVARIANCE : List[List[int]] = [[.5, .4], [.4, .5]]

data_1_multivariate : np.array = np.random.multivariate_normal(
    DATA_1_MULTIVARIATE_MEAN,
    DATA_1_COVARIANCE,
    size = DATA_1_SIZE,
)

data_2_multivariate : np.array = np.random.multivariate_normal(
    DATA_2_MULTIVARIATE_MEAN,
    DATA_2_COVARIANCE,
    size = DATA_2_SIZE,
)

plt.figure(figsize=(10, 6))

plt.scatter(data_1_multivariate[:, 0 ], data_1_multivariate[:, 1] )
plt.scatter(data_2_multivariate[:, 0], data_2_multivariate[:, 1])

sns.kdeplot(x= data_1_multivariate[:, 0], y= data_1_multivariate[:, 1],
            levels=15, linewidth=10, color='k', alpha=0.2)
sns.kdeplot(x=data_2_multivariate[:, 0], y=data_2_multivariate[:, 1],
            levels=15, linewidth=10, color="r")
plt.grid(False)
plt.show()

#%% GMM 1D 
N_SAMPLES : int = 100
MU1, SIGMA_1 = -5, 1.2
MU2, SIGMA_2 = 5, 1.8
MU3, SIGMA_3 = 0, 1.6

x1 : np.array = np.random.normal(loc = MU1, scale= np.sqrt(SIGMA_1), size = N_SAMPLES) 
x2 : np.array = np.random.normal(loc = MU2, scale= np.sqrt(SIGMA_2), size = N_SAMPLES) 
x3 : np.array = np.random.normal(loc = MU3, scale= np.sqrt(SIGMA_3), size = N_SAMPLES) 

x_con = np.concatenate((x1, x2, x3))
def plot_probabilityDensityFunc1D(mu : float, std :float, label, alpha : float = 0.5,
                                linestyle ='k--', density : bool =True,
                                color: str="green",
                                ) -> None:
    """
        Plot 1-D and its PDF
    Args:
        mu (float): _description_
        variance (float): _description_
        alpha (float, optional): _description_. Defaults to 0.5.
        linestyle (str, optional): _description_. Defaults to 'k--'.
        density (bool, optional): _description_. Defaults to True.
        color (str, optional): _description_. Defaults to "green".
    """
    X = norm.rvs(mu, std, size=1000) ## Draw random samples from a given distribution

    plt.hist(X, bins = 50, density=density, alpha=alpha, label=label, color=color)

    # plot pdf
    x = np.linspace(X.min(), X.max(), 1000)
    y = norm.pdf(x, mu, std)
    plt.plot(x, y, linestyle)

plot_probabilityDensityFunc1D(MU1,SIGMA_1,label=r"$\mu={} \ ; \ \sigma={}$".format(MU1,SIGMA_1),color=None)
plot_probabilityDensityFunc1D(MU2,SIGMA_2,label=r"$\mu={} \ ; \ \sigma={}$".format(MU2,SIGMA_2),color=None)
plot_probabilityDensityFunc1D(MU3,SIGMA_3,label=r"$\mu={} \ ; \ \sigma={}$".format(MU3,SIGMA_3),color=None)
plt.title("Original Distribution")
plt.legend()
plt.show()
    
#%% STEP 01 :Initilaize Mean, Cov, and weights
def random_initializer(x_con, n_clasters : int) -> Tuple[List[float], List[float], List[float]]:
    
    pi : np.array = np.ones((n_clasters)) / n_clasters
    means : List[float] = np.random.choice(x_con, n_clasters)
    variances : List[float] = np.random.random_sample(size=n_clasters)
    plot_probabilityDensityFunc1D(means[0], variances[0], 'Random init dist 01')
    plot_probabilityDensityFunc1D(means[1], variances[1], 'Random init dist 02', color="blue")
    plot_probabilityDensityFunc1D(means[2], variances[2], 'Random init dist 03', color="orange")
    
    plt.title('Random intialization')
    plt.legend()
    plt.show()
    
    return means, variances, pi
   
def expectation_step(x_con, n_clasters : int,
                     means : List[float], variances : List[float]):
    weights : np.array = np.zeros((n_clasters, len(x_con)))
    for j in range(n_clasters):
        weights[j, :] = norm(loc=means[j], scale=np.sqrt(variances[j])).pdf(x_con)
    return weights

def maximization_step(x_con, weights,
                      means : List[float], variances : List[float],
                      n_clasters : List[int], pi):

    r : List[np.array] = []

    for j in range(n_clasters):
        numerator : np.array = weights[j] * pi[j] 
        denominator : np.array = np.sum([weights[i] * pi[i] for i in range (n_clasters)], axis=0) ## 
        r.append(numerator / denominator)

        newMeanNumerator : np.array = np.sum(r[j] * x_con)
        newMeanDenominator : np.array = np.sum(r[j])
        means[j] : List[float] = newMeanNumerator/newMeanDenominator 

        newVariancesNumerator : np.array = np.sum(r[j] * np.square(x_con - means[j]))
        newVariancesDenominator : np.array = np.sum(r[j])
        variances[j] = newVariancesNumerator / newVariancesDenominator

        newPi = np.mean(r[j]) 
        pi[j] = newPi
        
    return means, variances, pi

def plot_steps(means : List[float ], variances : List[float],):
    plot_probabilityDensityFunc1D(MU1, SIGMA_1, alpha=0.0, linestyle='r--', label="Original Distribution")
    plot_probabilityDensityFunc1D(MU2, SIGMA_3, alpha=0.0, linestyle='r--', label="Original Distribution")
    plot_probabilityDensityFunc1D(MU3, SIGMA_3, alpha=0.0, linestyle='r--', label="Original Distribution")
    
    color_gen : Tuple[str] = (x for x in['green', 'blue', 'orange'])

    for mu, sigma in zip(means, variances):
        plot_probabilityDensityFunc1D(mu, sigma,
                                      alpha=0.5, label='d',
                                      color=next(color_gen))
    plt.title('Plotting steps')
    plt.show()

def train_gmm(data, n_clasters : int = 3,
              n_steps : int = 50, is_plot_steps = True):
    
    means, variances, pi = random_initializer(data,n_clasters)
    for step in range(n_steps):
        weights  = expectation_step(data, n_clasters, means, variances)
        means, variances, pi = maximization_step(data, weights,
                                                 means, variances,
                                                 n_clasters, pi)     
        if is_plot_steps:
            plot_steps(means, variances)
    plot_steps(means, variances)
 
train_gmm(x_con, )   

# %%
