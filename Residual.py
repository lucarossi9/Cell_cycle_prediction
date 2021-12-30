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
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, AveragePooling1D, Flatten, Conv1DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def residual_block (x , downsample = True, units = 10, reg = 1e-4, kernel_size = 3, dropout_rate = 0.1):

    y = layers.Dropout(dropout_rate)(x)
    y = keras.layers.Dense(units=units,
                            kernel_regularizer=keras.regularizers.l2(reg),
                            kernel_initializer=keras.initializers.glorot_normal(seed=None),
                            )(y)
    relu = ReLU()(y)
    y = BatchNormalization()(relu)
    y = layers.Dropout(dropout_rate)(y)
    y = keras.layers.Dense(units=units,
                            kernel_regularizer=keras.regularizers.l2(reg),
                            kernel_initializer=keras.initializers.glorot_normal(seed=None),
                            )(y)

    if downsample:
        x = layers.Dropout(dropout_rate)(x)
        x = keras.layers.Dense(units=units,
                                kernel_regularizer=keras.regularizers.l2(reg),
                                kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                )(x)

    out = Add()([x, y])
    relu = ReLU()(out)
    bn = BatchNormalization()(relu)
    return bn


def encoder(latent_dim, name = 'encoder', reg = 1e-2, size = [3, 3, 3], training = True, dropout_rate = 0.1):
    """sets up the encoder model"""
    
    def func(x):
        if training:
            x = layers.Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        else:
            x = BatchNormalization()(x)
        x = keras.layers.Dense(units=240,
                                kernel_regularizer=keras.regularizers.l2(reg),
                                kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                )(x)
        relu = ReLU()(x)
        x = BatchNormalization()(relu)
        
        for i in range(len(size)):
            w = size[i]
            x = residual_block(x, downsample=True, units=w)
        
        x = keras.layers.Dense(name=name + '_out',
                                units=1,
                                use_bias=False,
                                #kernel_regularizer=keras.regularizers.l2(reg),
                                kernel_initializer=keras.initializers.glorot_normal(seed=None)
                                )(x)  
        return x
    return func

def circular_unit(name, comp = 2):
    
    def func(x):
        out = []
        if comp < 2:
            raise ValueError("comp must be at least 2")
        elif comp == 2:
            out = [keras.layers.Lambda(lambda x: x+keras.backend.pow(keras.backend.sin(x), 2), name=name + '_sin')(x),
                  keras.layers.Lambda(lambda x: x+keras.backend.pow(keras.backend.cos(x), 2), name=name + '_cos')(x)]
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
                               kernel_initializer=keras.initializers.glorot_normal(seed=None),
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
                               kernel_initializer=keras.initializers.glorot_normal(seed=None),
                               bias_initializer=keras.initializers.Zeros()
                               )(x)
        return x

    return func



def linear_bypass(name , n , reg):
    
    def func(x):
        x = keras.layers.Dense(name=name + '_out',
                               units=n,
                               use_bias=False,
                               kernel_regularizer=keras.regularizers.l2(reg),
                               kernel_initializer=keras.initializers.glorot_normal(seed=None)
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

# transforming the data
dataset = "pc3" 
filepath = '/Statistical_computation/project4/data/McDavid'
data, cpt = load_data(dataset, filepath)
data = data.to_numpy()
data = preprocessing.scale(data)



rate = 2e-3
input_width = data.shape[1]
encoder_depth = 2
encoder_size = [30, 20]
n_circular_unit= 2
n_logistic_unit= 0
n_linear_unit= 2
n_linear_bypass= 3
dropout_rate = 0
nonlinear_reg = 1e-4
linear_reg = 1e-4


y = keras.Input(shape=(input_width,), name='input')
#x = encoder(name = 'encoder', reg = 1e-4, latent_dim = 1, size = [64, 32, 16, 16, 12], training = True, dropout_rate = 0.1)(y)
x = encoder(name = 'encoder', reg = 1e-3, latent_dim = 1, size = [32,  16, 8, 4, 2], training = True, dropout_rate = 0.05)(y)


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

my_callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs', write_grads = True)
]
model.fit(data, data, epochs=500, verbose=1, callbacks=my_callbacks)

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

##################
# CHLA9
##################
data = pd.read_csv(filepath+"/Statistical_computation/project4/final_dataset.csv")
data = data.iloc[: , 1:]
data = data.to_numpy()
data = preprocessing.scale(data)

rate = 5e-4
input_width = data.shape[1]
n_circular_unit= 2
n_logistic_unit= 2
n_linear_unit=  3
n_linear_bypass= 5
dropout_rate = 0.1
nonlinear_reg = 1e-4
linear_reg = 1e-4

y = keras.Input(shape=(input_width,), name='input')
x = encoder(name = 'encoder', reg = 1e-4, latent_dim = 1, size = [2000, 960, 480, 240, 120, 60, 30, 20, 10, 5, 3], training = True, dropout_rate = 0.1)(y)

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
model.fit(data, data, epochs=30, verbose=1)

pseudotime = keras.backend.function(inputs=[model.get_layer('input').input],
                                     outputs=[model.get_layer('encoder_out').output]
                                     )([data])[0]

pseudotime = (pseudotime % (2 * np.pi)) / 2
label = np.array([0 for p in pseudotime if p > 2.5] )
import seaborn as sns
sns.distplot(pseudotime, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                  label = pseudotime)


################
# Hyperparameters
################
import keras_tuner as kt
import tensorflow as tf

def model_builder(hp):

  units = hp.Choice('units', values=[2, 3, 4, 5, 6, 7])
  w = np.array([2**u for u in range(units)])
  rate = hp.Choice('learning_rate', values=[2e-4, 1e-3, 1e-4, 5e-4])
  input_width = data.shape[1]
  n_circular_unit= 2
  n_logistic_unit= hp.Int('logistic', min_value=0, max_value=3, step=1)
  n_linear_unit= hp.Int('linear', min_value=0, max_value=3, step=1)
  n_linear_bypass= 3
  dropout_rate = hp.Choice('rate', values=[0.05, 0.1, 0.2])
  nonlinear_reg = 1e-4
  linear_reg = 1e-4
  y = keras.Input(shape=(input_width,), name='input')
  x = encoder(name = 'encoder', reg = 1e-4, latent_dim = 1, size = w, training = True, dropout_rate = dropout_rate)(y)

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
data, cpt = load_data(dataset, filepath)
data = data.to_numpy()
data = preprocessing.scale(data)

tuner = kt.Hyperband(hypermodel = model_builder,
                      objective = kt.Objective("mean_squared_error", direction="min"),
                      max_epochs = 1000)
tuner.search_space_summary() 
tuner.search(data, data)
best_hp=tuner.get_best_hyperparameters()[0]
h_model = tuner.hypermodel.build(best_hp)
h_model.summary()
h_model.fit(data, data)
pseudotime = keras.backend.function(inputs=[h_model.get_layer('input').input],
                                      outputs=[h_model.get_layer('encoder_out').output]
                                      )([data])[0]
flat_embedding = (pseudotime % (2 * np.pi)) / 2
width = 3.14 / 100 / 2;
discrete_time, distr_g0g1 = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g0/g1', 0])
discrete_time, distr_s = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='s', 0])
discrete_time, distr_g2m = cyclum.evaluation.periodic_parzen_estimate(flat_embedding[np.squeeze(cpt)=='g2/m', 0])
correct_prob = cyclum.evaluation.precision_estimate([distr_g0g1, distr_s, distr_g2m], cpt, ['g0/g1', 's', 'g2/m'])
dis_score = correct_prob
print("Score: ", dis_score)
