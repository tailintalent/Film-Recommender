import recoEngine
import numpy as np
import random

def loadRatingsFromFileTest():
    filename = './ml-100k/u.data'
    print recoEngine.loadRatingsFromFile(filename)[1]

# loadRatingsFromFileTest()

def cofiCostFuncTest():
    num_users = 2
    num_movies = 2
    num_features = 2
    params = np.array([1.04868550176651, -0.400231959974807, 0.780851230996702, -0.385625911150141,
                       0.285443615432104, -1.68426508574498, 0.505013214197751, -0.454648461044761])
    Y = np.array([[5, 4], [3, -10]])
    R = np.array([[1, 1], [1, 0]])
    lambda_reg = 1.5

    print recoEngine.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_reg)



def cofiCostFuncGradTest():
    num_users = 2
    num_movies = 2
    num_features = 2
    params = np.array([ 1.32084098, -1.65723768,  0.69094855, -0.86692606,  1.20180086, -1.50788602,
  0.88187943, -1.1064699 ])
    Y = np.array([[5, 4], [3, -10]])
    R = np.array([[1, 1], [1, 0]])
    lambda_reg = 1.5

    grads = recoEngine.cofiCostFuncGrad(params, Y, R, num_users, num_movies, num_features, lambda_reg)
    print(grads)

# cofiCostFuncGradTest()

def collabFilteringTest():
    filename = './ml-100k/u.data'
    num_features = 10
    lambda_reg = 1.5

    print(" Reading ratings from {0} ...".format(filename))
    args = recoEngine.loadRatingsFromFile(filename) + (num_features, lambda_reg)
    num_users = args[2]
    num_movies = args[3]

    # prepare initialization values for X and Theta
    params_length = num_users*num_features + num_movies*num_features
    params = np.array([random.random() for _ in xrange(params_length)])

    # run training
    res = recoEngine.collabFiltering(filename, params, args)
    print "Optimized preference vector and feature vector: {0}".format(res)
    return res

collabFilteringTest()