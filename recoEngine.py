import numpy as np
from scipy import optimize

def loadRatingsFromFile(filename):
    file = open(filename, 'r')
    num_users = 943
    num_movies = 1682
    Y = np.zeros((num_movies, num_users))
    R = np.zeros((num_movies, num_users))
    for line in file:
        line_split = line.split("\t")
        usr_idx = int(line_split[0])-1
        movies_idx = int(line_split[1])-1
        rating = int(line_split[2])
        Y[movies_idx][usr_idx] = rating
        R[movies_idx][usr_idx] = 1

    return (Y, R, num_users, num_movies)


def cofiCostFunc(params, *args): # may use y_row and r_row instead to decouple
    (Y, R, num_users, num_movies, num_features, lambda_reg) = args
    # extract data for X and Theta from params
    X_raw = params[:(num_features*num_movies)]
    Theta_raw = params[(num_features*num_movies):]
    # reshape X and Theta
    X = np.reshape(X_raw, (num_movies, num_features))
    Theta = np.reshape(Theta_raw, (num_users, num_features))

    # transpose Theta
    Theta_t = np.transpose(Theta)

    # Y_pred
    Y_pred = np.dot(X, Theta_t)

    # Y is defined say double for real scores, -10 for empty entries
    Error = (Y_pred-Y)*R
    error_cost = np.sum(Error*Error)/2

    # regularization term cost
    X_cost = lambda_reg * np.sum(X*X)/2
    Theta_cost = lambda_reg * np.sum(Theta*Theta)/2

    # total cost
    total_cost = error_cost + X_cost + Theta_cost
    print "Current total cost is: {0}".format(total_cost)

    return total_cost

def cofiCostFuncGrad(params, *args):
    (Y, R, num_users, num_movies, num_features, lambda_reg) = args
    # extract data for X and Theta from params
    X_raw = params[:(num_features*num_movies)]
    Theta_raw = params[(num_features*num_movies):]
    # reshape X and Theta
    X = np.reshape(X_raw, (num_movies, num_features))
    Theta = np.reshape(Theta_raw, (num_users, num_features))

    # transpose Theta
    Theta_t = np.transpose(Theta)

    # Y_pred
    Y_pred = np.dot(X, Theta_t)

    # X_grad
    X_grad = np.dot((Y_pred - Y)*R, Theta) + lambda_reg*X
    Theta_grad = np.dot((np.transpose(Y_pred)- np.transpose(Y))*np.transpose(R), X) + lambda_reg*Theta

    return np.append(X_grad.ravel(), Theta_grad.ravel())

def collabFiltering(filename, params, args):
    res = optimize.fmin_cg(cofiCostFunc, params, fprime=cofiCostFuncGrad, args=args, maxiter=1000)
    return res


