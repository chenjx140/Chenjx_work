#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2 as cv
import os
import math
# Do not import any more packages than the above
'''
    La1 1 Assignment 
    Based on Python Introduction Notes: https://github.com/dmorris0/python_intro

    Complete the following functions by replacing the pass command and/or variables set equal to None
    Functions need to return the specified output.  In most cases only a single line of code is required.  
    To test your functions and get a score out of 20, call:
      python lab1_student_score.py
    Or run lab1_score.py in VSCode.  When you have everything correct, the output should be:
....................
----------------------------------------------------------------------
Ran 20 tests in 0.100s

OK
    Also, you'll see 3 images displayed which you can close.
'''

####################
# Chapter 4: Strings

def find_warning(message: str) -> str:    
    '''
    Returns the index of the first instance of the substring "warning" or "Warning" (or any other variation on the capitalization)
        If there is no "warning" substring, returns -1
    '''
    ind = int(-1)
    lo = message.lower()
    ind = lo.find("warning")
    return ind

def every_third(message: str) -> str:
    '''
    Returns every third letter in message starting with the second letter
    '''
    strf = ""
    for i in range(1,len(message),3):
        strf +=message[i]
    return strf


def all_words(message: str) -> str:
    '''
    Breaks message up at each space (" ") and puts the substrings into a list in the same order
    (Don't worry about punctuation)
    '''
    listf = message.split(" ")
    return listf
    
def half_upper_case(message: str) -> str:
    '''
    Returns new_message, where new_message has the same letters as message, but the first half
        of the letters are upper case and the rest lower case.  
        If there are an odd number of letters, round down, that is the first half will have one fewer letters
    '''
    
    str_f = message[0:len(message)//2]
    str_e = message[len(message)//2:]
    str_fin = str_f.upper() + str_e
    return str_fin

#############################
# Chapter 5: Numbers and Math

def c_to_f(degree_c: float) -> float:    
    '''
    Converts Celcius to Fahrenheit using the formula
    °F = °C * 9/5 + 32 
    Returns output in Fahrenheit
    '''
    return degree_c*9/5+32
    
    
def exp_div_fun(a: int, b: int, c: int) -> int:
    '''
    Return the integer remainder you get when you multiply a times itself b times and then divide by c
    '''
    return math.pow(a,b)%c
    
 
 #################################
# Chapter 6: Functions and Loops
    
    
def lcm(x: int, y: int) -> int:
    '''
    @@@@@@@@@@@
    Return lowest common multiple of x and y
    Method: let m be the larger of x and y
    Let testval start at m and in a loop increment it by m while testval is not divisible by both x and y
    return testval
    Hint: % will be useful
    '''
    m = max(x,y)
    ls = int(0)
    ls += m
    
    while( ls%x!=0 or (ls%y!=0)):
       ls = ls+m
    return ls


##################################################
# Chapter 8: Conditional Logic and Control Flow

def cond_cum_sum(a: int, b: int) -> int:
    '''
    Find the cumulative sum of numbers from 0 to a-1 that are not divisible by b
    Hint: % will be useful
    '''
    sum = 0
    ind = 0
    while(ind <= (a-1)):
        if(ind % b != 0):
            sum += ind
        ind+=1
    return sum

def divide_numbers(a: float, b: float) -> float:
    ''' 
    Return a / b
    Perform exception handling for ZeroDivisionError, and in this
    case return signed infinity that is the same sign as a
    Hint: np.sign() and np.inf will be useful
    '''
    if( b == 0):
        if(np.sign(a)  >= 0):
            return np.inf
        else:
            return -np.inf
    else:
        return a/b

##################################################
# Chapter 9: Tuples, Lists and Dictionaries    

def inc_list(a: int, b: int) -> list:
    '''
    Return a list of numbers that start at a and increment by 1 to b-1
    '''
    in_lis = []
    for i in range(a,b):
        in_lis.append(i)
    return in_lis

def make_all_lower( string_list: list ) -> list:
    ''' Use a single line of Python code for this function
        string_list: list of strings
        returns: list of same strings but all in lower case
        Hint: list comprehension
    '''
    return [i.lower() for i in string_list]

def decrement_string(mystr: str) -> str:
    ''' Use a single line of Python code for this function (hint: list comprehension)
        mystr: a string
        Return a string each of whose characters has is one ASCII value lower than in mystr
        Hint: ord() returns ASCII value, chr() converts ASCII to character, join() combines elements of a list
    '''
    str1=""
    return str1.join([chr(ord(x)-1) for x in mystr])

def list2dict( my_list: list ) -> dict:
    ''' 
    Return a dictionary corresponding to my_list where the keys are elements of my_list
    and the values are the square of the key
    '''
    dic = {}
    for i in my_list:
        dic[i] = pow(i,2)
    return dic


def concat_tuples( tuple1: tuple, tuple2: tuple ) -> tuple:
    ''' 
    Return a tuple that concatenates tuple2 to the end of tuple1
    '''
    return tuple1 + tuple2


##################################################
# Chapter 13: Mathematical Tools    
    
def matrix_multiplication(A: np.array,B: np.array) -> np.array:
    ''' 
    A, B: numpy arrays
    Return: matrix multiplication of A and B
    '''
    return np.dot(A,B)

def largest_row(M: np.array) -> np.array:
    ''' 
    M: 2D numpy array
    Return: 1D numpy array corresponding to the row with the greatest sum in M
    Hint: use np.argmax
    '''
    m = np.sum(M,axis = 1)
    i = np.argmax(m,axis= 0)
    return M[i]

def column_scale( A: np.array, vec: np.array) -> np.array:
    '''
    A: [M x N] 2D array
    vec: lenth N array
    return [M x N] 2D array where the i'th column is the corresponding column of A * vec[i]
    Hint: use broadcasting to do this in a single line
    '''
    return np.array(A * vec)
    

def row_add( A: np.array, vec: np.array) -> np.array:
    '''
    A: [M x N] 2D array
    vec: lenth M array
    return [M x N] 2D array where the i row is the corresponding row of A + vec[i]
    Hint: use broadcasting to do this in a single line
    '''
    return np.array(A.T + vec).T

## Test some OpenCV functions
class BlueBall:

    def __init__(self, impath: str):
        self.impath = impath
        self.img = cv.imread(impath)      
        if self.img is None:
            print('')
            print('*'*60)  # If we can't find the image, check impath
            print('** Current folder is:      ',os.getcwd())
            print('** Unable to read image:   ',impath)  
            print('** Pass correct path to data folder during initialization')
            self.init_succeed = False
        else:
            self.init_succeed = True

    def canny_edge(self, t1: int=100, t2: int=200, show=False) -> np.array:
        ''' t1, t2: two threshold for Canny Edge detection on image
            show: if True, then displays canny edges in a window called 'Canny'
            Hint: use the cv.Canny() function
        '''
        edges = cv.Canny(self.img,t1,t2)   # Replace this
        if show:
            cv.imshow('Canny',edges)
        return edges

    def find_ball_pix(self, show=False) -> np.array:
        ''' Returns a binary mask image.  This should be 1 if the red channel is < 100, and zero otherwise
            show: if True, then shows the mask in a window called 'Ball Pix'
        '''
     
        #print(ball_pix)
        _, ball_pix = cv.threshold(self.img[:,:,2],100-1,1, cv.THRESH_BINARY_INV)

        if show:
            cv.imshow('Ball Pix',ball_pix.astype(np.uint8)*255)
        return ball_pix

    def centroid_of_ball(self, show=False) -> tuple:
        ''' Find centroid of mask pixels 
            center_xy: (x,y) tuple of center
            show: if True, then displays a color image called 'Centroid' and plots a red circle at the centroid
            Hint: a simply way to find centroids is using the mask and the np.where() function
        '''
        pass
        _, ball_pix = cv.threshold(self.img[:,:,2],100-1,1, cv.THRESH_BINARY_INV)
        M = cv.moments(ball_pix)
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

        # the logic from https://blog.csdn.net/tengfei461807914/article/details/76626631

       
        center_xy = (x,y)  # Replace this
        if show:
            outim = self.img.copy()
            icenter_xy = tuple([int(i) for i in center_xy])
            cv.circle(outim, icenter_xy, 5, (0,0,255), -2)
            cv.imshow('Centroid', outim)
        return center_xy



if __name__=="__main__":

    # A simple way to debug your functions above is to create tests below.  Then
    # you can run this code directly either in VSCode or by typing: 
    # python lab1_review.py

    # For example, you could do
    print( find_warning("Here is a warning") )
