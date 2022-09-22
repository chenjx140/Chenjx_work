# Team work 
# Peizeng Kang
# Jinxian Cheng
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

def main():
    data = io.loadmat('./data.mat')
    train_X = data["X"][:150, :]
    train_y = data["Y"][:150, :]
    test_X = data["X"][150:, :]
    test_y = data["Y"][150:, :]
   
    para_c =  [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 200]  
    kernel_func = ['linear', 'poly', 'rbf', 'sigmoid']
    tot = []
    for i in kernel_func:
        func_with_diff_c = []
        for j in para_c:
            svm_svc = svm.SVC(C=j, kernel=i)
            svm_svc.fit(train_X, train_y.ravel())
            func_with_diff_c.append(len(svm_svc.support_vectors_))
        tot.append(func_with_diff_c)
        
    plt.plot(para_c, tot[0], label=kernel_func[0], color="blue")
    plt.plot(para_c, tot[1], label=kernel_func[1], color="yellow")
    plt.plot(para_c, tot[2], label=kernel_func[2], color="red")
    plt.plot(para_c, tot[3], label=kernel_func[3], color="green")
    plt.xlabel("C value")
    plt.ylabel("Support Vector Number")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Question on 4
    # C is the regularization parameter which is L2 penalty.
    # It must be greater than 0. It means once the number of
    # c is greater, the penalty for those misclassify points
    # is greater then the boundary is more thinner. In our 
    # graph, it was shown than the number of support_vector
    # decrease. 
    # For the same C number, penalty aparameter, the different 
    # kernel function have different number of support vectors 
    # at C = 25, the sequence of support vector number is 
    # "rbf" > "poly" > "sigmoid" > "linear". 
    # When C = 100, the sequence is "poly" > "rbf" > "linear" > "sigmoid"     
            
            

if __name__ == "__main__":
    main()