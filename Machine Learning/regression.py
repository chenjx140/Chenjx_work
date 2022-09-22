# CSE 404 Intro to Machine Learning
# Homework 5: Linear Regression & Optimization

# imports
import time
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Randomly split the dataset into training & testing sets
# @train_perc :: Percentage (in decimal format) of data to use for training
#                Example: if train_perc == 0.7 --> 70% training, 30% testing
def rand_split_train_test(data, label, train_perc):
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    data_tr = data[:num_train_sample]
    data_te = data[num_train_sample:]

    label_tr = label[:num_train_sample]
    label_te = label[num_train_sample:]

    return data_tr, data_te, label_tr, label_te


# Takes a subsample of the entire dataset
def subsample_data(data, label, subsample_size):
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])
    data, label = shuffle(data, label)
    data = data[:subsample_size]
    label = label[:subsample_size]
    return data, label


# Generates a random dataset with dimensions based on feature_size & sample_size
# @bias :: for Gaussian noise
def generate_rnd_data(feature_size, sample_size, bias=False):
    # Generate X matrix
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)

    # Generate ground truth model
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10

    # Generate labels
    label = np.dot(data, truth_model)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model



# Sine Function :)
def sine_data(sample_size, order_M, plot_data = False, noise_level = 0.1, bias = False):
    if int(order_M) != order_M: 
        raise Exception('order_M should be an integer.')
    if order_M < 0:
        raise Exception('order_M should be at least larger than 0.')
    
    # Generate X matrix
    x = np.random.rand(sample_size,1) * 2 * np.pi        # generate x from 0 to 2pi
    X = np.column_stack([ x**m for m in range(order_M)])

    data = np.concatenate((X, np.ones((sample_size, 1))), axis=1) if bias else X

    # Ground truth model: a sine function
    f = lambda x: np.sin(x)

    # Generate labels
    label = f(x)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)*noise_level

    if plot_data:
        plt.figure()
        xx = np.arange(0, np.pi * 2, 0.001)
        yy = f(xx)
        plt.plot(xx, yy, linestyle = '-', color = 'g', label = 'Objective Value')
        plt.scatter(x, label, color = 'b', marker = 'o', alpha = 0.3)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Sine Data (N = %d) with Noise Level %.4g.".format(sample_size, noise_level))
        plt.show()

    return data, label, f


######################################################################################

def mean_squared_error(true_label, predicted_label):
    """
        Compute the mean square error between the true and predicted labels
        :param true_label: Nx1 vector
        :param predicted_label: Nx1 vector
        :return: scalar MSE value
    """
    
    mse = np.sqrt(np.sum((true_label -predicted_label ) **2)/true_label.size)
    return mse



def least_squares(feature, target):
    """
        Compute the model vector obtained after MLE
        w_star = (X^T X)^(-1)X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :return: w_star (d+1)x1 model vector
        """
    w_star = np.dot(np.linalg.inv(np.dot(feature.T,feature)),np.dot(feature.T,target))
    return w_star



def ridge_regression(feature, target, lam = 1e-17):
    """
        Compute the model vector when we use L2-norm regularization
        w_star = (X^T X + lambda I)^(-1) X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :param lam: the scalar regularization parameter, lambda
        :return: w_star (d+1)x1 model vector
        """
    feature_dim = feature.shape[1]
    I = np.eye(feature_dim)

    w_star = np.dot(np.linalg.inv(np.dot(feature.T,feature) + I*lam),np.dot(feature.T,target))
    return w_star



# K-fold cross validation
def k_fold(current_fold, data, total_sample_size,label):
    #TODO
    #return data_train, data_test, label_train, label_test
    size = int(total_sample_size/4)


    subsample_size = np.max([1, np.min([data.shape[0], size])])
    data, label = shuffle(data, label)
    data1 = data[:subsample_size]
    label1 = label[:subsample_size]

    data2 = data[subsample_size:subsample_size*2]
    label2 = label[subsample_size:subsample_size*2]

    data3 = data[subsample_size*2:subsample_size*3]
    label3 = label[subsample_size*2:subsample_size*3]

    data4 = data[subsample_size*3:subsample_size*4]
    label4 = label[subsample_size*3:subsample_size*4]

    
    data_train = []
    data_test = []
    label_train = []
    label_test = []

    if(current_fold == 1):
        #data_test = data2+data3+data4
        data_train = np.vstack((data2,data3))
        data_train =  np.vstack((data_train, data4))
        data_test = data1
        #label_train = label2+label3+label4
        label_train = np.vstack((label2,label3))
        label_train = np.vstack((label_train,label4))
        label_test = label1

    if(current_fold == 2):
        #data_train = data1+data3+data4
        data_train = np.vstack((data1,data3))
        data_train =  np.vstack((data_train, data4))

        data_test = data2
        #label_train = label1+label3+label4

        label_train = np.vstack((label1,label3))
        label_train = np.vstack((label_train,label4))
        label_test = label2
    if(current_fold == 3):
        #data_train = data1+data2+data4
        data_train = np.vstack((data1,data2))
        data_train =  np.vstack((data_train, data4))

        data_test = data3
        #label_train = label1+label2+label4

        label_train = np.vstack((label1,label2))
        label_train = np.vstack((label_train,label4))
        label_test = label3
    if(current_fold == 4):
        #data_train = data1+data2+data3
        data_train = np.vstack((data1,data2))
        data_train =  np.vstack((data_train, data3))

        data_test = data4
        #label_train = label1+label2+label3
        label_train = np.vstack((label1,label2))
        label_train = np.vstack((label_train,label3))
        label_test = label4

    return data_train, data_test, label_train, label_test

    


########################################################################################

def compute_gradient(feature, target, model, lam = 1e-17):
    #w_star = (X^T X + lambda I)^(-1) X^T t
    # Compute the gradient of linear regression objective function with respect to w
   
    grad = feature.T.dot(feature.dot(model)) - (feature.T.dot(target)) +lam*model
    #Lr = np.dot(feature.T,(np.dot(feature,model) - target[:,0]))
    #G = Lr+lam*model
    #return gradient
    return grad




# Gradient Descent
def gradient_descent(feature, target, step_size, max_iter, lam = 1e-17):
    model_o = np.zeros(feature.shape[1])
    lst = []
    ther = 1
    for i in range(max_iter):
        # Compute gradient
        gt = compute_gradient(feature,target,model_o,lam = 1e-17)
        # Update the model
        model_o = model_o - step_size*(gt)
        # Compute the error (objective value)
        trn = mean_squared_error(target,np.dot(feature, model_o))
        lst.append(trn)
        if i >= 1:
            ther = abs(lst[i] -lst[i-1])
        if (ther <= 1e-5):
            break
  
    #return #model, objective value
    return model_o, np.array(lst)

def plot_objective_function(objective_value, i = 0):

    plt.figure()
    plt.plot(objective_value)
    if i != 0:
        plt.title(" the gradient descent optimizers SGD n:" + str(i))
    else:
        plt.title(" the gradient descent optimizers GD ")
    plt.ylabel("Objective value")
    plt.xlabel("iterations")
    
    plt.show()

# Stochastic Gradient Descent
def batch_gradient_descent(feature, target, step_size, max_iter, batch_size, lam = 1e-17):
   
    model_o = np.zeros(feature.shape[1])
    lst = []
    ther = 1
    for i in range(max_iter):
        feature,target = subsample_data(feature, target,batch_size)
        # Compute gradient
        gt = compute_gradient(feature,target,model_o,lam = 1e-17)
        # Update the model
        model_o = model_o - step_size*(gt)
        # Compute the error (objective value)
        trn = mean_squared_error(target,np.dot(feature, model_o))
        lst.append(trn)
        if i >= 1:
            ther = abs(lst[i] -lst[i-1])
        if (ther <= 1e-5):
            break
  
    return model_o, np.array(lst)

# Plots/Errors
# def plot_objective_function(objective_value, batch_size=None):
# def print_train_test_error(train_data, test_data, train_label, test_label, model):

##########################################################################################

# TODO: Homework Template
if __name__ == '__main__':
    #plt.interactive(False)
    #np.random.seed(491)
    
    # Problem 1
    # Complete Least Squares, Ridge Regression, MSE
    # Randomly generate & plot 30 data points using sine function
    # Randomly split the dataset
	# For each lambda, use Ridge Regression to calculate & plot MSE for training & testing sets
	# Implement k-fold CV & choose best lambda
    #plt.figure()
    data, label, f = sine_data(30,10,True,noise_level = 0.3)
    train_perc = 0.7
    data_tr, data_te, label_tr, label_te = rand_split_train_test(data, label, train_perc)

    list_of_lam = [1e-1, 1e-5,1e-2,1e-1,1,10,100,1000]
    
    plot_y1 = []
    plot_y2 = []
    for i in list_of_lam:
        reg_m_r = ridge_regression(data_tr,label_tr,i)

        trn = mean_squared_error(label_tr,np.dot(data_tr, reg_m_r))
        tes = mean_squared_error(label_te,np.dot(data_te, reg_m_r))
        plot_y1.append(trn)
        plot_y2.append(tes)

    plt.xlabel("lamda number")
    plt.ylabel("error")
    plt.plot([1,2,3,4,5,6,7,8],plot_y1,'ro')
    plt.plot([1,2,3,4,5,6,7,8],plot_y2,'bs')
    plt.show()


    data, label, f = sine_data(32,10,False)

    #k_fold(1, data, 32,label)


    average = []
    for i in list_of_lam:
        error = 0
        for j in range(1,5):
            data_tr, data_te, label_tr, label_te = k_fold(j, data, 32,label)
            reg_m_r = ridge_regression(data_tr,label_tr,i)

            trn = mean_squared_error(label_tr,np.dot(data_tr, reg_m_r))
            tes = mean_squared_error(label_te,np.dot(data_te, reg_m_r))

            error += (abs(trn)+abs(tes))/2
        error = error/4
        average.append(error)
    print(average)

    min_lam = min(average)

    min_lam_f = average.index(min_lam)

    print(list_of_lam[min_lam_f])

    data, label,f = generate_rnd_data(50, 1000)
    #data, label, _ = sine_data(1000, 50, plot_data = True, noise_level = 0.1, bias = False)
            
    m,o = gradient_descent(data, label, 0.0001, 1000)


    plot_objective_function(o)

    for i in [5,10,100,500]:

        m,x = batch_gradient_descent(data, label, 0.0001, 1000, i, lam = 1e-17)

        plot_objective_function(x,i)

    # Problem 2
    # Complete Gradient Descent & Stochastic GD
    # Implement ridge regression with GD & plot objectives at each iteration
    # Implement SGD & plot objectives at each iteration per batch 

	
    

    
    
