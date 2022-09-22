# Homework 2: Perceptron

import numpy as np  #functions, vectors, matrices, linear algebra...
from random import choice   #randomly select item from list
from random import seed
from random import random
from random import randint
import matplotlib.pyplot as plt #plots

def train_perceptron(training_data, plot = None):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)    #np.random.rand(model_size)
    iteration = 1
    while True:
        # compute results according to the hypothesis
        resu = np.sign(np.multiply(np.matmul(X,w),y))
        # get incorrect predictions (you can get the indices)
        i = np.arange(X.shape[0])
        misc_i = i[resu != 1]
        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if len(misc_i) == 0:

            if plot:
                plot(w)
            break
        # Pick one misclassified example.
        pick_r = choice(misc_i)
        x_s , y_s = X[pick_r],y[pick_r]
        # Update the weight vector with perceptron update rule
        w += y_s *x_s 
        iteration += 1

    return w , iteration # for Q3
def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))


#the function from the lecture
def compute_y(w,x):
    if w[1] == 0:
        print("divided by 0")
        return 0
    return -(w[2]+w[0]*x)/w[1]


# the function from the lecture
def plot_decision_boundary(w, X, y, x_min=-1, x_max=1, y_min=-1, y_max=1, target = False):
    pos_points = X[y ==  1, :2]
    neg_points = X[y == -1, :2]

    plt.figure(1, figsize = (10, 8))

    x_limit = (x_min - 0.5, x_max + 0.5)
    y_limit = (y_min - 0.5, y_max + 0.5)

    plt.plot(pos_points[:, 0], pos_points[:, 1], "bo", markersize=15)
    plt.plot(neg_points[:, 0], neg_points[:, 1], "rx", markersize=15)

    line_x = np.linspace(x_limit[0], x_limit[1], 50)
    line_y = compute_y(w, line_x)
    
    x_range = min(x_min,y_min,x_max,y_max)
    y_range = max(x_max,y_max,x_min,y_min)
    target = np.arange(x_range,y_range)

    plt.plot(line_x, line_y, linestyle='-', color='k', linewidth=3)
    plt.plot(target, 2*target, linestyle='-', color='b', linewidth=3)
    plt.xlim(x_limit[0], x_limit[1])
    plt.ylim(y_limit[0], y_limit[1])

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()
    return

if __name__ == '__main__':
    
    rnd_x = np.array([[0,1,1],\
                      [0.6,0.6,1],\
                      [1,0,1],\
                      [1,1,1],\
                      [0.3,0.4,1],\
                      [0.2,0.3,1],\
                      [0.1,0.4,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    trained_model , _ = train_perceptron(rnd_data)
    print("Model:", trained_model)
    print_prediction(trained_model, rnd_x)


    
    # Q3 part:
    # ramdam generator
    #idea: https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/

    #a,b

    rnd_x = np.array([[1,3,1],\
                      [2,5,1],\
                      [3,7,1],\
                      [4,9,1],\
                      [10,21,1],\
                      [1,9,1],\
                      [2,5,1],\
                      [0.2,1,1],\
                      [2,4.1,1],\
                      [2,11,1],\
                      [0.6,0.2,1],\
                      [1,0.5,1],\
                      [1,1.3,1],\
                      [3.5,0.6,1],\
                      [2.4,4.6,1],\
                      [3.9,1,1],\
                      [5.2,0.6,1],\
                      [1.9,0.3,1],\
                      [1,1.2,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    plot_fuc = lambda model: plot_decision_boundary(model, rnd_data[0], rnd_data[1], min(rnd_data[0][:, 0]), max(rnd_data[0][:, 0]), 
                        min(rnd_data[0][:, 1]), max(rnd_data[0][:, 1]))
    trained_model , itera = train_perceptron(rnd_data, plot_fuc)
    print(itera)
        #trained_model = train_perceptron(rnd_data)
    print("Mode2:", trained_model)
    print_prediction(trained_model, rnd_x)
    





    # c,d,e
    seed(1)
    rep = [20,100,1000]
    for reep in rep:
        n = reep
        x = np.random.random((n,3))
        for i in range(n):
            x[i][2] =1
            x[i][1] *= randint(1,10)
            x[i][0] *= randint(1,10)
        #print(x)
        # target function: 
        # y = x
        
        list_y = []
        for i in range(n):
            #print("n:",target[0]*x[i][0]+ target[1])
            if 2*x[i][0]  <= x[i][1]:
                list_y.append(1)
            else:
                list_y.append(-1)
        list_y = np.array(list_y)


        rnd_data = [x,list_y]

        
        plot_fuc = lambda model: plot_decision_boundary(model, rnd_data[0], rnd_data[1], min(rnd_data[0][:, 0]), max(rnd_data[0][:, 0]), 
                        min(rnd_data[0][:, 1]), max(rnd_data[0][:, 1]))
        trained_model , itera = train_perceptron(rnd_data, plot_fuc)
        print(itera)
        #trained_model = train_perceptron(rnd_data)
        print("Mode2:", trained_model)
        print_prediction(trained_model, x)




    



