import recoEngine
import numpy as np
import random

def loadMoviesIntoDBFromFileTest():
    filename = './ml-1m/movies.dat'
    recoEngine.loadMoviesIntoDBFromFile(filename)

# loadMoviesIntoDBFromFileTest()

def loadUsersIntoDBFromFileTest():
    filename = './ml-1m/users.dat'
    recoEngine.loadUsersIntoDBFromFile(filename)

# loadUsersIntoDBFromFileTest()

def loadRatingsIntoDBFromFileTest():
    filename = './ml-1m/ratings.dat'
    recoEngine.loadRatingsIntoDBFromFile(filename)

# loadRatingsIntoDBFromFileTest()

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
    num_features = 10
    lambda_reg = 1.5

    print(" Loading ratings from database ...")
    args = recoEngine.readRatingsFromDB() + (num_features, lambda_reg)
    num_users = args[2]
    num_movies = args[3]

    # prepare initialization values for X and Theta
    params_length = num_users*num_features + num_movies*num_features
    params = np.array([random.random() for _ in xrange(params_length)])

    # run training
    (X, Theta) = recoEngine.collabFiltering(params, args)
    print "Updating X and Theta into database..."
    recoEngine.updateXandThetaIntoDB(X, Theta)

# collabFilteringTest()

def updateXandThetaIntoDBTest():
    X = np.array([
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3]
    ])

    Theta = np.array([
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3]
    ])

    recoEngine.updateXandThetaIntoDB(X, Theta)

# updateXandThetaIntoDBTest()

if __name__ == '__main__':
    # set-up database from files
    loadMoviesIntoDBFromFileTest()
    loadUsersIntoDBFromFileTest()
    loadRatingsIntoDBFromFileTest()

    # run reco algo
    collabFilteringTest()