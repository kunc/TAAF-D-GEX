
import pandas as pd
import numpy as np
import os.path as osp
import os


from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import Nadam
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, CSVLogger


# set seed for better repeatability
from numpy.random import seed as set_seed
from tensorflow import set_random_seed
import random

seed = 42
set_seed(seed)
set_random_seed(seed)
random.seed = seed

data_path = './example_data'
fname_train_y = osp.join(data_path, 'ddf_tg_dataset_small_example_train.h5')
fname_test_y = osp.join(data_path, 'ddf_tg_dataset_small_example_test.h5')


fname_train_X = osp.join(data_path, 'ddf_lm_dataset_small_example_train.h5')
fname_test_X = osp.join(data_path, 'ddf_lm_dataset_small_example_test.h5')


n_epoch = 600  # number of epochs
batch_size = 64
l_rate = 5e-4


def lr_schedule(epoch: int) -> float:
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    Parameters
    ----------
    epoch : int
        The number of epochs

    Returns
    -------
    lr : float32
        learning rate
    """
    lr = 5e-4
    if epoch > 575:
        lr *= 0.5e-3
    elif epoch > 550:
        lr *= 1e-3
    elif epoch > 475:
        lr *= 1e-2
    elif epoch > 400:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


# load the data
ddf_train_y = pd.read_hdf(fname_train_y, 'table')
ddf_test_y = pd.read_hdf(fname_test_y, 'table')

ddf_train_X = pd.read_hdf(fname_train_X, 'table')
ddf_test_X = pd.read_hdf(fname_test_X, 'table')

# for checking the input dimensions
in_size = ddf_train_X.shape[1]
out_size = ddf_train_y.shape[1]
print('Input size: {}  Output size: {}'.format(in_size, out_size))


# define the model
n_hidden = 2  # number of hidden layers
n_neurons = 300  # number of neurons in each hidden layer
activation_function_name = 'sigmoid'
dropout = 0.25  # 25 % of neurons won't be used each training epoch
model_name = 'classical_demonstrator_small'

inputs = Input(shape=(in_size,), name='inputs')
x = inputs
for n in range(n_hidden):
    x = Dense(n_neurons, activation=activation_function_name, name='H' + str(n))(x)
    x = Dropout(dropout, seed=seed)(x)

outputs = Dense(out_size, activation='linear', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)
optimizer = Nadam(lr=l_rate)
model.compile(optimizer=optimizer, loss='mean_absolute_error', )
model.summary()

# set up training
log_dir = f'./logs/{model_name}/'
os.makedirs(log_dir, exist_ok=True)
model_dir = './models/{}/'.format(model_name)
os.makedirs(model_dir, exist_ok=True)

csv_logger = CSVLogger(osp.join(log_dir, f'{model_name}_training_log.csv'), append=True)
tb = TensorBoard(log_dir=log_dir)
callbacks = [csv_logger, tb]

plot_model(model, osp.join(model_dir, f'{model_name}_visualization.png'), show_shapes=True, show_layer_names=True)

# train the model
model.fit(x=ddf_train_X.values,
          y=ddf_train_y.values,
          batch_size=batch_size,
          epochs=n_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=(ddf_test_X.values, ddf_test_y.values),
          shuffle=True,
          initial_epoch=0)

model.save(osp.join(model_dir, '{}_last_model.hdf5'.format(model_name)))