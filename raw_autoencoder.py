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
import gpflow
from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter

################
# Preprocessing
################
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


dataset = "pc3" 
filepath = '/Statistical_computation/project4/data/McDavid'
data, cpt = load_data(dataset, filepath)
data = data.to_numpy()
seed_value = 10        

###############
# MODEL
###############
def circular_unit(name, comp = 2):
    
    def func(x):
        out = []
        if comp < 2:
            raise ValueError("comp must be at least 2")
        elif comp == 2:
            out = [keras.layers.Lambda(lambda x: keras.backend.sin(x), name=name + '_sin')(x),
                   keras.layers.Lambda(lambda x: keras.backend.cos(x), name=name + '_cos')(x)]
            # out = [keras.layers.Lambda(lambda x: keras.backend.pow(keras.backend.sin(x), 2), name=name + '_sin')(x),
            #        keras.layers.Lambda(lambda x: keras.backend.pow(keras.backend.cos(x), 2), name=name + '_cos')(x)]
        else:
            out = [
                keras.layers.Lambda(lambda x: keras.backend.sin(x + 2 * pi * i / comp), name=name + '_' + str(i))(x)
                for i in range(comp)]
        out = keras.layers.Concatenate(name=name + '_out')(out)
        return out

    return func

def logistic_unit(name, n, trans = True, reg_scale = 1e-2, reg_trans = 1e-2):
    
    def func(x):
        x = keras.layers.Dense(name=name + '_scale',
                               units=n,
                               use_bias=trans,
                               kernel_regularizer=keras.regularizers.l2(reg_scale),
                               bias_regularizer=keras.regularizers.l2(reg_trans),
                               kernel_initializer=keras.initializers.glorot_normal(seed=seed_value),
                               bias_initializer=keras.initializers.Zeros()
                               )(x)
        x = keras.layers.Activation(name=name + '_out',
                                    activation='tanh'
                                    )(x)
        return x

    return func


def linear_unit(name, n, trans = True, reg_scale = 1e-2, reg_trans = 1e-2):
    
    def func(x):
        x = keras.layers.Dense(name=name + '_scale',
                               units=n,
                               use_bias=trans,
                               kernel_regularizer=keras.regularizers.l2(reg_scale),
                               bias_regularizer=keras.regularizers.l2(reg_trans),
                               kernel_initializer=keras.initializers.glorot_normal(seed=seed_value),
                               bias_initializer=keras.initializers.Zeros()
                               )(x)
        return x

    return func

def encoder(name , size , reg , drop , act = 'tanh'):
    
    def func(x):
        for i in range(len(size)):
            w = size[i]
            x = keras.layers.Dense(name=name + str(i) + '_scale',
                                   units=w,
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=seed_value),
                                   )(x)
            if drop > 0:
                x = keras.layers.Dropout(name=name + str(i) + '_dropout',
                                         rate=drop
                                         )(x)
            x = keras.layers.Activation(name=name + str(i) + '_act',
                                        activation=act
                                        )(x)

        x = keras.layers.Dense(name=name + '_out',
                               units=1,
                               use_bias=False,
                               #kernel_regularizer=keras.regularizers.l2(reg),
                               kernel_initializer=keras.initializers.glorot_normal(seed=seed_value)
                               )(x)
        return x

    return func


def linear_bypass(name , n , reg):
    
    def func(x):
        x = keras.layers.Dense(name=name + '_out',
                               units=n,
                               use_bias=False,
                               kernel_regularizer=keras.regularizers.l2(reg),
                               kernel_initializer=keras.initializers.glorot_normal(seed=seed_value)
                               )(x)
        return x

    return func


def decoder(name, n):

    def func(x: list):
        if len(x) > 1:
            x = keras.layers.Concatenate(name=name + '_concat')(x)
        else:
            x = x[0]
        x = keras.layers.Dense(name=name + '_out',
                               units=n,
                               use_bias=False,
                               kernel_initializer=keras.initializers.Zeros()
                               )(x)
        return x

    return func

# H9
# rate = 2e-4
# input_width = data.shape[1]
# encoder_depth = 2
# encoder_size = [30, 20]
# n_circular_unit= 2
# n_logistic_unit= 0
# n_linear_unit= 0
# n_linear_bypass= 3
# dropout_rate = 0.1
# nonlinear_reg = 1e-4
# linear_reg = 1e-4

#pc3
# rate = 2e-4
# input_width = data.shape[1]
# encoder_depth = 2
# encoder_size = [30, 20]
# n_circular_unit= 2
# n_logistic_unit= 0
# n_linear_unit= 0
# n_linear_bypass= 3
# dropout_rate = 0.1
# nonlinear_reg = 1e-4
# linear_reg = 1e-4


#mb
rate = 2e-4
input_width = data.shape[1]
encoder_depth = 2
encoder_size = [30, 20]
n_circular_unit= 2
n_logistic_unit= 0
n_linear_unit= 0
n_linear_bypass= 3
dropout_rate = 0
nonlinear_reg = 1e-4
linear_reg = 1e-4


y = keras.Input(shape=(input_width,), name='input')
x = encoder('encoder', encoder_size, nonlinear_reg, dropout_rate, 'selu')(y)

chest = []
if n_linear_bypass > 0:
    x_bypass = linear_bypass('bypass', n_linear_bypass, linear_reg)(y)
    chest.append(x_bypass)
if n_logistic_unit > 0:
    x_logistic = logistic_unit('logistic', n_logistic_unit)(x)
    chest.append(x_logistic)
if n_linear_unit > 0:
    x_linear = linear_unit('linear', n_linear_unit)(x)
    chest.append(x_linear)
if n_circular_unit > 0:
    x_circular = circular_unit('circular')(x)
    chest.append(x_circular)
y_hat = decoder('decoder', input_width)(chest)
model = keras.Model(outputs=y_hat, inputs=y)
model.compile(loss='mean_squared_error',
                           optimizer=keras.optimizers.Adam(rate))

model.fit(data, data, epochs=1000, verbose=1)


pseudotime = keras.backend.function(inputs=[model.get_layer('input').input],
                                     outputs=[model.get_layer('encoder_out').output]
                                     )([data])[0]

flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score: ", dis_score)

################################
# DHLA 
###############################
data = pd.read_csv(filepath+"/Statistical_computation/project4/final_dataset.csv")
data = data.iloc[: , 1:]
data = data.to_numpy()
data = preprocessing.scale(data)

rate = 2e-4
input_width = data.shape[1]
encoder_depth = 2
encoder_size = [300, 150, 50, 20]
n_circular_unit= 2
n_logistic_unit= 0
n_linear_unit= 0
n_linear_bypass= 3
dropout_rate = 0
nonlinear_reg = 1e-4
linear_reg = 1e-4


y = keras.Input(shape=(input_width,), name='input')
x = encoder('encoder', encoder_size, nonlinear_reg, dropout_rate, 'selu')(y)

chest = []
if n_linear_bypass > 0:
    x_bypass = linear_bypass('bypass', n_linear_bypass, linear_reg)(y)
    chest.append(x_bypass)
if n_logistic_unit > 0:
    x_logistic = logistic_unit('logistic', n_logistic_unit)(x)
    chest.append(x_logistic)
if n_linear_unit > 0:
    x_linear = linear_unit('linear', n_linear_unit)(x)
    chest.append(x_linear)
if n_circular_unit > 0:
    x_circular = circular_unit('circular')(x)
    chest.append(x_circular)
y_hat = decoder('decoder', input_width)(chest)
model = keras.Model(outputs=y_hat, inputs=y)
model.compile(loss='mean_squared_error',
                           optimizer=keras.optimizers.Adam(rate))
model.fit(data, data, epochs=100, verbose=1)

pseudotime = keras.backend.function(inputs=[model.get_layer('input').input],
                                     outputs=[model.get_layer('encoder_out').output]
                                     )([data])[0]

flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score: ", dis_score)

###########################
# Hyperparameters tunning
##########################
import keras_tuner as kt

def model_builder(hp):

  w1 = hp.Int('units1', min_value=32, max_value=128, step=32)
  w2 = hp.Int('units2', min_value=16, max_value=64, step=16)
  w3 = hp.Int('units3', min_value=8, max_value=32, step=8)
  w4 = hp.Int('units4', min_value=4, max_value=16, step=4)
  rate = hp.Choice('learning_rate', values=[2e-4, 1e-3, 1e-4, 5e-4])
  input_width = data.shape[1]
  n_circular_unit= 2
  n_logistic_unit= hp.Int('units1', min_value=0, max_value=3, step=1)
  n_linear_unit= hp.Int('units1', min_value=0, max_value=3, step=1)
  n_linear_bypass= 3
  dropout_rate = hp.Choice('learning_rate', values=[0.05, 0.1, 0.2])
  nonlinear_reg = 1e-4
  linear_reg = 1e-4
  y = keras.Input(shape=(input_width,), name='input')
  x = encoder('encoder', [w1, w2, w3, w4], nonlinear_reg, dropout_rate, 'selu')(y)
  chest = []
  if n_linear_bypass > 0:
      x_bypass = linear_bypass('bypass', n_linear_bypass, linear_reg)(y)
      chest.append(x_bypass)
  if n_logistic_unit > 0:
      x_logistic = logistic_unit('logistic', n_logistic_unit)(x)
      chest.append(x_logistic)
  if n_linear_unit > 0:
      x_linear = linear_unit('linear', n_linear_unit)(x)
      chest.append(x_linear)
  if n_circular_unit > 0:
      x_circular = circular_unit('circular')(x)
      chest.append(x_circular)
  y_hat = decoder('decoder', input_width)(chest)
  model = keras.Model(outputs=y_hat, inputs=y)
  model.compile(loss='mean_squared_error',
                             optimizer=keras.optimizers.Adam(rate),
                             metrics=[
                            tf.keras.metrics.MeanSquaredError(),
                            ])
  return model

dataset = "mb" 
data, cpt = load_data(dataset)
data = data.to_numpy()
data = preprocessing.scale(data)

tuner = kt.Hyperband(hypermodel = model_builder,
                     objective = kt.Objective("mean_squared_error", direction="min"),
                     max_epochs = 100)
tuner.search_space_summary() 
tuner.search(data, data)
best_hp=tuner.get_best_hyperparameters()[0]
# h_model = tuner.hypermodel.build(best_hp)
# h_model.summary()
# h_model.fit(data, data)
# pseudotime = keras.backend.function(inputs=[h_model.get_layer('input').input],
#                                      outputs=[h_model.get_layer('encoder_out').output]
#                                      )([data])[0]
# flat_embedding = (pseudotime % (2 * np.pi)) / 2
# width = 3.14 / 100 / 2;
# discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
# discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
# discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
# correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
# dis_score = correct_prob
# print("Score: ", dis_score)
