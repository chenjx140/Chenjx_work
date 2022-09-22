import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from numpy.core.numeric import identity


train_data = (np.loadtxt('ZipDigits.train',delimiter=' ',dtype=str)[:,:-1]).astype(float)
test_data = (np.loadtxt('ZipDigits.test',delimiter=' ',dtype=str)[:,:-1]).astype(float)


ss = []
f1 = False
f2 = False
for i in range(0,15):
    data_row=train_data[i][1:]
    print(train_data[i][0])
    if(train_data[i][0] == 1.0) and f1 == False:
        pixels = np.matrix(data_row)
        
        pixels=pixels.reshape(16,16)
        ss = pixels
       
        plt.figure(figsize=(5,5))
        plt.imshow(pixels)
        plt.show()
        f1 = True
    if(train_data[i][0] == 5.0) and f2 == False:
        pixels = np.matrix(data_row)
        pixels=pixels.reshape(16,16)
        plt.figure(figsize=(5,5))
        plt.imshow(pixels)
        plt.show()

        f2 = True


def classify_smy(m):
    m_r = np.eye(16)[::-1]
    imageflip = m.dot(m_r)   
    imagesymm = abs(m - imageflip)
    averg = -1 * imagesymm.mean()
    return averg


def idensity(m):
    i = Counter(m)[-1.0]/256
    #return 1-i
    return i
    



x,y = train_data[:,1:],train_data[:,0]

feature_1 = []
feature_5 = []
x_1 = x[(y == 1)]
y_1 = y[(y == 1)]

x_5 = x[(y == 5)]
y_5 = y[(y == 5)]
for img in x_1:

    ide = idensity(img)
    
    pixels = np.matrix(img)
        
    pixels=pixels.reshape(16,16)

    smy = classify_smy(pixels)

    feature_1.append([ide,smy])

for img in x_5:

    ide = idensity(img)
    
    pixels = np.matrix(img)
        
    pixels=pixels.reshape(16,16)

    smy = classify_smy(pixels)

    feature_5.append([ide,smy])



feature_1 = np.array(feature_1)
feature_5 = np.array(feature_5)
plt.scatter(feature_1[:,0],feature_1[:,1],c='b',marker='o')
plt.scatter(feature_5[:,0],feature_5[:,1],c='r',marker='x')

plt.xlabel('Inversed Intensity')
plt.ylabel('Symmetry')

plt.title('classfy 1 and 5')

plt.show()

def compute_gradient(feature, target, model):
    lisC = []
    K = np.exp(np.dot(feature,model)*(target))+1

    for i in range(feature.shape[0]):
        lisC.append(-(target[i]*feature[i])/K[i])
    
    return np.mean(lisC,axis=0)



def cost_function(feature, target,model):
    c = np.log(np.exp(np.dot(feature,model)*(-target))+1)
    return np.mean(c)


def gradient_descent(feature, target, step_size, max_iter):

    model_o = np.zeros(feature.shape[1])

   
    lst = []
    for i in range(max_iter):
        # Compute gradient

        gt = compute_gradient(feature,target,model_o)
        # Update the model
        model_o = model_o - step_size*(gt)
        # Compute the error (objective value)

        
        cost = cost_function(feature, target,model_o)
        lst.append(cost)

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




ls = []
    
data_1 = []
data_2=[]
data_tr, data_te, label_tr, label_te = train_data[:,1:],train_data[:,0],test_data[:,1:],test_data[:,0]

data_tr = data_tr[(data_te==1)+(data_te==5)]
data_te =  data_te[(data_te==1)+(data_te==5)]

for img in data_tr:

    ide = idensity(img)
    
    pixels = np.matrix(img)
        
    pixels=pixels.reshape(16,16)

    smy = classify_smy(pixels)

    data_1.append([ide,smy])

data_1 = np.array(data_1)

data_te[data_te == 5] = -1 

Me, ls = gradient_descent(data_1, data_te, 2.5, 600)

label_tr = label_tr[(label_te==1)+(label_te==5)]
label_te =  label_te[(label_te==1)+(label_te==5)]

for img in label_tr:

    ide = idensity(img)
    
    pixels = np.matrix(img)
        
    pixels=pixels.reshape(16,16)

    smy = classify_smy(pixels)

    data_2.append([ide,smy])

data_2 = np.array(data_2)

label_te[label_te == 5] = -1 



plot_objective_function(ls, i = 0)




# learned from the https://stackoverflow.com/questions/41050906/how-to-plot-the-decision-boundary-of-logistic-regression-in-scikit-learn
x_min, x_max = data_2[:, 0].min() - .5, data_2[:, 0].max() + .5
y_min, y_max = data_2[:, 1].min() - .5, data_2[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
plt.subplot(1, 1, 1)


zz = np.dot(np.c_[xx.ravel(),yy.ravel()],Me)
zz[zz>0] = 1
zz[zz<=0] = -1
zz = zz.reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='RdBu', alpha=0.5)
plt.scatter(data_2[:, 0], data_2[:, 1], c=label_te, cmap='RdBu', linewidth=1)
plt.ylabel('symmetry', fontsize=14)
plt.xlabel('intensity', fontsize=14)
        
plt.xlim(xx.min(), xx.max())
plt.title('decision boundary')
plt.show()


Ein = cost_function(data_1, data_te,Me)
print("Ein : ",Ein)


test = cost_function(data_2, label_te,Me)
print("test : ",test)

