import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cyclum.tuning
import keras
from keras import layers
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import concatenate
from sklearn.model_selection import train_test_split
import cyclum.models
import cyclum.evaluation
import cyclum.illustration
################
# Preprocessing
################

ddef load_data(dataset, filepath):
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


dataset = "H9" 
filepath = '/Statistical_computation/project4/data/McDavid'
data, cpt = load_data(dataset, filepath)
data = data.to_numpy()
seed_value = 10        

# Train (H9)
model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=3, 
                                     epochs=2000, rate=5e-4, verbose=100,
                                     encoder_width=[30, 20])

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[30, 20], 
                                  encoder_depth=2,
                                 n_circular_unit=2,
                                 n_logistic_unit=0,
                                 n_linear_unit=2,
                                 n_linear_bypass=3,
                                 dropout_rate=0.1)

model.train(data, epochs=2000, verbose=100, rate=2e-4)
pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)


color_map = {"g0/g1": "red", "s": "green", "g2/m": "blue"}
cyclum.illustration.plot_round_distr_color(pseudotime[:, 0], cpt.squeeze(), color_map)
# MB
dataset = "mb" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)
model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=5, 
                                     epochs=2000, rate=5e-4, verbose=100,
                                      encoder_width=[30, 20])

model.train(data, epochs=1000, verbose=100, rate=2e-4)
pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)

pseudotime = (pseudotime % (2 * np.pi)) / 2
color_map = {"g0/g1": "red", "s": "green", "g2/m": "blue"}
cyclum.illustration.plot_round_distr_color(pseudotime[:, 0], cpt.squeeze(), color_map)

# PC3
dataset = "pc3" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)
# model = cyclum.tuning.CyclumAutoTune(data, max_linear_dims=5, 
#                                       epochs=2000, rate=5e-4, verbose=100,
#                                       encoder_width=[30, 20])

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[30, 20], 
                                  encoder_depth=2,
                                 n_circular_unit=2,
                                 n_logistic_unit=0,
                                 n_linear_unit=0,
                                 n_linear_bypass=3,
                                 dropout_rate=0.1)
model.train(data, epochs=1000, verbose=100, rate=2e-4)

pseudotime = model.predict_pseudotime(data)
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score Cyclum: ", dis_score)

pseudotime = (pseudotime % (2 * np.pi)) / 2
color_map = {"g0/g1": "red", "s": "green", "g2/m": "blue"}
cyclum.illustration.plot_round_distr_color(pseudotime[:, 0], cpt.squeeze(), color_map)


#####################
# CHLA 9
##################
data = pd.read_csv(filepath+"/Statistical_computation/project4/final_dataset.csv")
data = data.iloc[: , 1:]
data = data.to_numpy()
data = preprocessing.scale(data)

model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                  encoder_width=[600, 300, 200, 100, 50, 20], 
                                  encoder_depth=6,
                                  n_circular_unit=2,
                                  n_logistic_unit=0,
                                  n_linear_unit=5,
                                  n_linear_bypass=3,
                                  dropout_rate=0.1)
model.train(data, epochs=100, verbose=10, rate=2e-4)


pseudotime = model.predict_pseudotime(data)
pseudotime = (pseudotime % (2 * np.pi)) / 2
pseudotime = pseudotime[:, 0]
label = np.array([0 for p in pseudotime if p < 1.5] )
import seaborn as sns
sns.distplot(pseudotime, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = pseudotime)

# Hyperparameters tunning
#########################################################
def objective(params):
    rate = params['rate']
    input_width = data.shape[1]
    depth = params['depth']
    n_circular_unit= 2
    n_logistic_unit= 0
    n_linear_unit= params['linear']
    n_linear_bypass= 3
    drop = params['drop']
    nonlinear_reg = 1e-4
    linear_reg = 1e-4   
    w = [2**u for u in range(depth)]

    model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                      encoder_width=w, 
                                      encoder_depth=len(w),
                                      n_circular_unit=2,
                                      n_logistic_unit=0,
                                      n_linear_unit=n_linear_unit,
                                      n_linear_bypass=3,
                                     dropout_rate=drop)
    
    mod = model.train(data, epochs=10, verbose=10, rate=rate)
    cost = np.mean(mod.history['loss'])   
    return cost

from hyperopt import fmin, tpe, hp


space = dict([('rate',  hp.choice('rate', [1e-5, 1e-4, 4e-4, 1e-3])),
                    ('depth', hp.choice('depth', range(7, 14, 1))),
                    ('linear', hp.choice('linear', range(2, 6, 1))),                  
                    ('drop', hp.choice('drop', [0.15, 0.2, 0.05, 0.1]))
                    ])

best = fmin(fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30)

