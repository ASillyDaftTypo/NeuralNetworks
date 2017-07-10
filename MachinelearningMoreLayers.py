import tensorflow as tf
import numpy as np
import tensorflowvisu
import math
import os


#Load combined files

global unencrypted_data_train
global encrypted_data_train
global unencrypted_data_test
global encrypted_data_test

unencrypted_data_train = np.load('Unencryptedbinaryshake'+'.npy')
encrypted_data_train = np.load('Encryptedbinaryshake'+'.npy')
unencrypted_data_test = np.load('Unencryptedbinarywap'+'.npy')
encrypted_data_test = np.load('Encryptedbinarywap'+'.npy')







#start of model

X = tf.placeholder(tf.float32, [None, 16])
Y_ = tf.placeholder(tf.float32, [None, 16])
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)
#Nodes per layer
K = 100
L = 200
M = 300
N = 400

O = 500
P = 400
Q = 300
R = 200
S = 100

#Weights and biases
W1 = tf.Variable(tf.truncated_normal([16, K],stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L],stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M],stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N],stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, O],stddev=0.1))
B5 = tf.Variable(tf.zeros([O]))

W6 = tf.Variable(tf.truncated_normal([O, P],stddev=0.1))
B6 = tf.Variable(tf.zeros([P]))
W7 = tf.Variable(tf.truncated_normal([P, Q],stddev=0.1))
B7 = tf.Variable(tf.zeros([Q]))
W8 = tf.Variable(tf.truncated_normal([Q, R],stddev=0.1))
B8 = tf.Variable(tf.zeros([R]))
W9 = tf.Variable(tf.truncated_normal([R, S],stddev=0.1))
B9 = tf.Variable(tf.zeros([S]))
W10 = tf.Variable(tf.truncated_normal([S, 16],stddev=0.1))
B10 = tf.Variable(tf.zeros([16]))


#layers
Y1 =tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 =tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 =tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 =tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Y5 =tf.nn.relu(tf.matmul(Y4d, W5) + B5)
Y5d = tf.nn.dropout(Y5, pkeep)

Y6 =tf.nn.relu(tf.matmul(Y5d, W6) + B6)
Y6d = tf.nn.dropout(Y6, pkeep)
Y7 =tf.nn.relu(tf.matmul(Y6d, W7) + B7)
Y7d = tf.nn.dropout(Y7, pkeep)
Y8 =tf.nn.relu(tf.matmul(Y7d, W8) + B8)
Y8d = tf.nn.dropout(Y8, pkeep)
Y9 =tf.nn.relu(tf.matmul(Y8d, W9) + B9)
Y9d = tf.nn.dropout(Y9, pkeep)

Ylogits = tf.matmul(Y9d, W10) + B10
Y =tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1]), tf.reshape(W7, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1]), tf.reshape(W10, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1]), tf.reshape(B7, [-1]), tf.reshape(B8, [-1]), tf.reshape(B9, [-1]), tf.reshape(B10, [-1])], 0)

#I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
#It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)

datavis = tensorflowvisu.MnistDataVis()

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)





# init
init = tf.global_variables_initializer()

#sess.run(init)

#with tf.Session() as sess:

batch_size = 1000
epochs = 10

update_train_data = True
update_test_data = True

    #sess.run(init)
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(coord=coord)

#number_of_batches = int((number_of_items - 0.5) / batch_size)




sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):
    '''
    for epoch in range(epochs):
        
        for batch_number in range(number_of_batches):

            start_index = int(batch_number * batch_size)
            end_index = start_index + batch_size
            
            batch_X = unencrypted_data[start_index:end_index]
            batch_Y = encrypted_data[start_index:end_index]
    '''
    #[1000*i:1000*i+999]
    #[100*i:(100*i)+99]
    #for batch_number in range(1000):
    #PASS THE GODDAMN VARIABLES THROUGH

    batch_X = unencrypted_data_train[100*i:(100*i)+99]
    batch_Y = encrypted_data_train[100*i:(100*i)+99]
    batch_X2 = unencrypted_data_test[100*i:(100*i)+99]
    batch_Y2 = encrypted_data_test[100*i:(100*i)+99]
    

    #i = epoch
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
        

    # compute training values for visualisation
    if update_train_data:
        a, c, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
        datavis.append_training_curves_data(i, a, c)
        #datavis.update_image1(im)
        datavis.append_data_histograms(i, w, b)

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X2, Y_: batch_Y2, pkeep: 1.0})
        print(str(i) + ": ** epoch " + str(i*100) + " ** test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        #datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})

datavis.animate(training_step, iterations=100000, train_data_update_freq=1, test_data_update_freq=1, more_tests_at_start=True)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

