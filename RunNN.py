import numpy as np
import MachinelearningNoGraph as ML

epochs=100
layer_width=64
layer_depth=2

act_function=0
optimiser_function=0

learning_rate_max=0.003
learning_rate_min=0.001
learning_rate_decay=2000

batch_quantity=700
batch_size=1000

unencrypted_data_train = np.load('Unencryptedbinaryshake32'+'.npy')
encrypted_data_train = np.load('Encryptedbinaryshake32'+'.npy')
unencrypted_data_test = np.load('Unencryptedbinarywap32'+'.npy')
encrypted_data_test = np.load('Encryptedbinarywap32'+'.npy')


ML.Run_NN(unencrypted_data_train, encrypted_data_train, unencrypted_data_test, encrypted_data_test,
       epochs, layer_width, layer_depth,
       act_function, optimiser_function,
       learning_rate_max, learning_rate_min, learning_rate_decay,
       batch_quantity, batch_size)
