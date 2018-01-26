

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2

np.random.seed(1)

'''

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()



index = 6
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))



X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


'''

def create_placeholders(n_H0, n_W0, n_C0, n_y):
 
    X = tf.placeholder(tf.float32,shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder(tf.float32,shape=(None,n_y))
    ### END CODE HERE ###None
    
    return X, Y




def initialize_parameters():
  
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer =  tf.contrib.layers.xavier_initializer(seed = 0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters




def forward_propagation(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, 4,activation_fn=None)
    ### END CODE HERE ###

    return Z3




def compute_cost(Z3, Y):
  
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    ### END CODE HERE ###
    
    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
  
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3        
                                     # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]   

    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            
            _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, temp_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(temp_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters









image=cv2.imread('/home/anand-ros/Desktop/cnn/b1.jpg')
image1=cv2.imread('/home/anand-ros/Desktop/cnn/b2.jpg')
image2=cv2.imread('/home/anand-ros/Desktop/cnn/b3.jpg')
image3=cv2.imread('/home/anand-ros/Desktop/cnn/b4.jpg')
image4=cv2.imread('/home/anand-ros/Desktop/cnn/b5.jpg')
image5=cv2.imread('/home/anand-ros/Desktop/cnn/b6.jpg')
image6=cv2.imread('/home/anand-ros/Desktop/cnn/b7.jpg')
image7=cv2.imread('/home/anand-ros/Desktop/cnn/b8.jpg')
image8=cv2.imread('/home/anand-ros/Desktop/cnn/c1.jpg')
image9=cv2.imread('/home/anand-ros/Desktop/cnn/c2.jpg')
image10=cv2.imread('/home/anand-ros/Desktop/cnn/c3.jpg')
image11=cv2.imread('/home/anand-ros/Desktop/cnn/c4.jpg')
image12=cv2.imread('/home/anand-ros/Desktop/cnn/c5.jpg')
image13=cv2.imread('/home/anand-ros/Desktop/cnn/c6.jpg')
image14=cv2.imread('/home/anand-ros/Desktop/cnn/c7.jpg')
image15=cv2.imread('/home/anand-ros/Desktop/cnn/c8.jpg')
image16=cv2.imread('/home/anand-ros/Desktop/cnn/c9.jpg')
image17=cv2.imread('/home/anand-ros/Desktop/cnn/c10.jpg')
image18=cv2.imread('/home/anand-ros/Desktop/cnn/l1.jpg')
image19=cv2.imread('/home/anand-ros/Desktop/cnn/l2.jpg')
image20=cv2.imread('/home/anand-ros/Desktop/cnn/l3.jpg')
image21=cv2.imread('/home/anand-ros/Desktop/cnn/l4.jpg')
image22=cv2.imread('/home/anand-ros/Desktop/cnn/l5.jpg')
image23=cv2.imread('/home/anand-ros/Desktop/cnn/l6.jpg')
image24=cv2.imread('/home/anand-ros/Desktop/cnn/l7.jpg')
image25=cv2.imread('/home/anand-ros/Desktop/cnn/l8.jpg')
image26=cv2.imread('/home/anand-ros/Desktop/cnn/l9.jpg')
image27=cv2.imread('/home/anand-ros/Desktop/cnn/r1.jpg')
image28=cv2.imread('/home/anand-ros/Desktop/cnn/r2.jpg')
image29=cv2.imread('/home/anand-ros/Desktop/cnn/r3.jpg')
image30=cv2.imread('/home/anand-ros/Desktop/cnn/r4.jpg')
image31=cv2.imread('/home/anand-ros/Desktop/cnn/r5.jpg')
image32=cv2.imread('/home/anand-ros/Desktop/cnn/r6.jpg')
image33=cv2.imread('/home/anand-ros/Desktop/cnn/r10.jpg')
image34=cv2.imread('/home/anand-ros/Desktop/cnn/r8.jpg')
image35=cv2.imread('/home/anand-ros/Desktop/cnn/r9.jpg')


arr=np.reshape(image,image1.shape[0]*image1.shape[1]*3)
arr1=np.reshape(image1,image1.shape[0]*image1.shape[1]*3)
arr2=np.reshape(image2,image2.shape[0]*image2.shape[1]*3)
arr3=np.reshape(image3,image3.shape[0]*image3.shape[1]*3)
arr4=np.reshape(image4,image3.shape[0]*image3.shape[1]*3)
arr5=np.reshape(image5,image3.shape[0]*image3.shape[1]*3)
arr6=np.reshape(image6,image3.shape[0]*image3.shape[1]*3)

arr7=np.reshape(image7,image.shape[0]*image.shape[1]*3)
arr8=np.reshape(image8,image1.shape[0]*image1.shape[1]*3)
arr9=np.reshape(image9,image2.shape[0]*image2.shape[1]*3)
arr10=np.reshape(image10,image3.shape[0]*image3.shape[1]*3)
arr11=np.reshape(image11,image3.shape[0]*image3.shape[1]*3)
arr12=np.reshape(image12,image3.shape[0]*image3.shape[1]*3)
arr13=np.reshape(image13,image3.shape[0]*image3.shape[1]*3)

arr14=np.reshape(image14,image.shape[0]*image.shape[1]*3)
arr15=np.reshape(image15,image1.shape[0]*image1.shape[1]*3)
arr16=np.reshape(image16,image2.shape[0]*image2.shape[1]*3)

arr17=np.reshape(image17,921600)
arr18=np.reshape(image18,image3.shape[0]*image3.shape[1]*3)
arr19=np.reshape(image19,image3.shape[0]*image3.shape[1]*3)
arr20=np.reshape(image20,image3.shape[0]*image3.shape[1]*3)


arr21=np.reshape(image21,image.shape[0]*image.shape[1]*3)
arr22=np.reshape(image22,image1.shape[0]*image1.shape[1]*3)
arr23=np.reshape(image23,image2.shape[0]*image2.shape[1]*3)
arr24=np.reshape(image24,image3.shape[0]*image3.shape[1]*3)
arr25=np.reshape(image25,image3.shape[0]*image3.shape[1]*3)
arr26=np.reshape(image26,image3.shape[0]*image3.shape[1]*3)
arr27=np.reshape(image27,image3.shape[0]*image3.shape[1]*3)

arr28=np.reshape(image28,image.shape[0]*image.shape[1]*3)
arr29=np.reshape(image29,image1.shape[0]*image1.shape[1]*3)
arr30=np.reshape(image30,image2.shape[0]*image2.shape[1]*3)
arr31=np.reshape(image31,image3.shape[0]*image3.shape[1]*3)
arr32=np.reshape(image32,image3.shape[0]*image3.shape[1]*3)
arr33=np.reshape(image33,image3.shape[0]*image3.shape[1]*3)
arr34=np.reshape(image34,image3.shape[0]*image3.shape[1]*3)
arr35=np.reshape(image35,image3.shape[0]*image3.shape[1]*3)

X1=np.asmatrix([arr,arr1,arr2,arr3,arr4,arr5,arr6,arr7])
X2=np.asmatrix([arr8,arr9,arr10,arr11,arr12,arr13,arr14,arr15,arr16,arr17])
X3=np.asmatrix([arr18,arr19,arr20,arr21,arr22,arr23,arr24,arr25,arr26])
X4=np.asmatrix([arr27,arr28,arr29,arr30,arr31,arr32,arr33,arr34,arr35])


Xtrain= np.array([image,image1,image2,image8,image9,image10,image18,image19,image20,image27,image28,image29])



Xtrain = Xtrain/255.0
Y1=np.asmatrix([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1]]).T
############################################################################
#Give input here
X_test=np.asarray([image,image1,image2,image3,image4,image5,image6,image7,image8,image9,image10,image11,image12,image13,image14,image15,image16,image17,image18,image19,image20,image21,image22,image23,image24,image25,image26,image27,image28,image29,image30,image31,image32,image33,image34,image35])
X_test=X_test/255.0
#######################################################################[0,1,0,0],[0,1,0,0]#####
Y1=Y1.T
print X_test.shape
Y_test=np.asmatrix([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]).T
Y_test=Y_test.T

parameters = model(Xtrain, Y1, X_test, Y_test,0.01,20)
















