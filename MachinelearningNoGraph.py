import tensorflow as tf
import numpy as np
import math

def Run_NN(unencrypted_data_train, encrypted_data_train, unencrypted_data_test, encrypted_data_test,
           epochs=100, layer_width=16, layer_depth=2,
           act_function= 0, optimiser_function= 0,
           learning_rate_max=0.003, learning_rate_min=0.001, learning_rate_decay=2000,
           batch_quantity=700, batch_size=100):
    
    #Load combined files
    '''
    global unencrypted_data_train
    global encrypted_data_train
    global unencrypted_data_test
    global encrypted_data_test
    '''
    unencrypted_data_train = np.load('Unencryptedbinaryshake32'+'.npy')
    encrypted_data_train = np.load('Encryptedbinaryshake32'+'.npy')
    unencrypted_data_test = np.load('Unencryptedbinarywap32'+'.npy')
    encrypted_data_test = np.load('Encryptedbinarywap32'+'.npy')

    bytearraysize = len(unencrypted_data_train[0])

    X = tf.placeholder(tf.float32, [None, bytearraysize])
    Y_ = tf.placeholder(tf.float32, [None, bytearraysize])
    lr = tf.placeholder(tf.float32)
    pkeep = tf.placeholder(tf.float32)
    
    W = {}
    B = {}

    W[1] = tf.Variable(tf.truncated_normal([bytearraysize, layer_width],stddev=0.1))
    B[1] = tf.Variable(tf.zeros([layer_width])/10)
    
    for layer in range(layer_depth):
        W[layer+2] = tf.Variable(tf.truncated_normal([layer_width, layer_width],stddev=0.1))
        B[layer+2] = tf.Variable(tf.zeros([layer_width])/10)

    W[layer_depth+2] = tf.Variable(tf.truncated_normal([layer_width, bytearraysize],stddev=0.1))
    B[layer_depth+2] = tf.Variable(tf.zeros([bytearraysize]))
    

    Y_ax = {}
    Y_ax_drop = {}

    
    Y_ax[1] =tf.nn.relu(tf.matmul(X, W[1]) + B[1])
    Y_ax_drop[1] = tf.nn.dropout(Y_ax[1], pkeep)

    for layer in range(layer_depth):
        Y_ax[layer+2] = tf.nn.relu(tf.matmul(Y_ax[layer+1], W[layer+2]) + B[layer+2])
        Y_ax_drop[layer+2] = tf.nn.dropout(Y_ax[layer+2], pkeep)

    Ylogits = tf.matmul(Y_ax_drop[layer_depth+1], W[layer_depth+2]) + B[layer_depth+2]
    Y =tf.nn.sigmoid(Ylogits)
    Y_clipped = tf.clip_by_value(Y, 1e-10, 0.999999)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_*tf.log(Y_clipped) + (1-Y_)*tf.log(1-Y_clipped), reduction_indices=1))

    correct_prediction = tf.equal(tf.round(Y_clipped) ,Y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    '''
    # matplotlib visualisation
    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1])], 0)#, tf.reshape(W3, [-1])], 0)#), tf.reshape(W4, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1])], 0)#), tf.reshape(W7, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1]), tf.reshape(W10, [-1])], 0)
    allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1])], 0)#, tf.reshape(B3, [-1])], 0)#), tf.reshape(B4, [-1]), tf.reshape(B5, [-1]), tf.reshape(B6, [-1])], 0)#), tf.reshape(B7, [-1]), tf.reshape(B8, [-1]), tf.reshape(B9, [-1]), tf.reshape(B10, [-1])], 0)
    '''

    # training step, the learning rate is a placeholder
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    accuracy_max = []
    
    # init
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    def training_step(epochs):
        accuracy_max = 0
        for i in range(epochs):
            a_train = []
            a_test = []
            c_train = []
            c_test = []

            for batch_number in range(batch_quantity):
                
                batch_X = unencrypted_data_train[batch_size*batch_number:(batch_size*(batch_number+1))-1]
                batch_Y = encrypted_data_train[batch_size*batch_number:(batch_size*(batch_number+1))-1]
                batch_X2 = unencrypted_data_test[batch_size*batch_number:(batch_size*(batch_number+1))-1]
                batch_Y2 = encrypted_data_test[batch_size*batch_number:(batch_size*(batch_number+1))-1]
                
                # learning rate decay
                learning_rate = learning_rate_min + (learning_rate_max - learning_rate_min) * math.exp(-i/learning_rate_decay)
                    

                # compute training values for visualisation

                a, c, output, original = sess.run([accuracy, cross_entropy, Y_clipped, Y_], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
                if Y_clipped[0] == Y_clipped[1] and Y_clipped[0] == Y_clipped[2] and Y_clipped[0] == Y_clipped[3]:
                        print("\n\nERROR: Batch Values Same! \n\n")
                        
                a_train.append(a)
                c_train.append(c)
                    

                # compute test values for visualisation

                a, c = sess.run([accuracy, cross_entropy], {X: batch_X2, Y_: batch_Y2, pkeep: 1.0})

                a_test.append(a)
                c_test.append(c)
         
                
                # the backpropagation training step
                sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})

            train_accuracy_average = sum(a_train)/batch_quantity
            train_entropy_average = sum(c_train)/batch_quantity
            test_accuracy_average = sum(a_test)/batch_quantity
            test_entropy_average = sum(c_test)/batch_quantity

            print(str(i) + " TRAIN: Accuracy: " + str(train_accuracy_average) + " Cross entropy: " + str(train_entropy_average))
            print(str(i) + "  TEST: Accuracy: " + str(test_accuracy_average) + " Cross entropy: " + str(test_entropy_average) + "\n")

            if test_accuracy_average > accuracy_max:
                  accuracy_max = test_accuracy_average

    training_step(epochs)


    print("max test accuracy: " + str(accuracy_max))
    return
