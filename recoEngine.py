import numpy as np
from scipy import optimize

def loadMoviesIntoDBFromFile(filename):
    # load movies from ml-1m to db
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client.mlMovieDb
    # set up the collection
    movie = db.movie
    movie.remove({})

    # start reading file
    file = open(filename, 'r')
    movie_idx = 0
    for line in file:
        line_split = line.split("::")
        movie_id = line_split[0]
        movie_name = unicode(line_split[1])
        genres = line_split[2].split("|")
        # since movie_id is not continuous, movie_idx is added to help calculation in Y and R
        movie_document = {"_id": movie_id, "movie_idx": movie_idx, "movie_name": movie_name, "genres": genres, "x":[1]*10}

        movie.save(movie_document)
        movie_idx += 1

    print "{0} movies in database.".format(movie.count())
    client.close()

def loadUsersIntoDBFromFile(filename):
    # load users from ml-1m to db
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client.mlMovieDb
    # set up the collection
    user = db.user
    user.remove({})

    file = open(filename, 'r')
    user_idx = 0
    for line in file:
        line_split = line.split("::")
        user_id = line_split[0]
        flag = 0 # 0 means old users
        user_document = {"_id": user_id, "user_idx": user_idx, "flag": flag, "theta":[1]*10}

        user.save(user_document)
        user_idx += 1

    print "{0} users in database.".format(user.count())
    client.close()

def loadRatingsIntoDBFromFile(filename):
    # load ratings from ml-1m to db
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client.mlMovieDb
    # set up the collection
    ratings = db.ratings
    ratings.remove({})

    file = open(filename, 'r')
    for line in file:
        line_split = line.split("::")
        user_id = line_split[0]
        movie_id = line_split[1]
        rating = int(line_split[2])
        timestamp = line_split[3]
        rating_document = {"_id": user_id + '::' + movie_id, "rating": rating, "timestamp":timestamp}

        ratings.save(rating_document)

    print "{0} ratings in database.".format(ratings.count())
    client.close()

def readRatingsFromDB():
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client.mlMovieDb
    # set up the collection
    ratings = db.ratings.find()
    users = db.user.find()
    movies = db.movie.find()
    num_users = db.user.count()
    num_movies = db.movie.count()

    # userDict and movieDict
    userDict = {}
    movieDict = {}
    for user in users:
        userDict[user["_id"]] = user["user_idx"]
    for movie in movies:
        movieDict[movie['_id']] = movie['movie_idx']

    Y = np.zeros((num_movies, num_users))
    R = np.zeros((num_movies, num_users))

    print Y.shape
    for rating in ratings:
        ids = rating["_id"]
        ids_split = ids.split("::")
        user_id = ids_split[0]
        movie_id = ids_split[1]

        # get user_idx and movie_idx from
        # user_id and movie_id respectively
        user_idx = userDict[user_id]
        movie_idx = movieDict[movie_id]

        # create Y and R
        Y[movie_idx][user_idx] = rating["rating"]
        R[movie_idx][user_idx] = 1

    client.close()
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

def collabFiltering(params, args):
    res = optimize.fmin_cg(cofiCostFunc, params, fprime=cofiCostFuncGrad, args=args, maxiter=1000)
    num_users = args[2]
    num_movies = args[3]
    num_features = len(params)/(num_movies + num_users)
    X_raw = res[:(num_features*num_movies)]
    Theta_raw = res[(num_features*num_movies):]
    X = np.reshape(X_raw, (num_movies, num_features))
    Theta = np.reshape(Theta_raw, (num_users, num_features))
    return (X, Theta)

def updateXandThetaIntoDB(X, Theta):
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    db = client.mlMovieDb

    # set up the collection
    movie_collection = db.movie
    user_collection = db.user

    (num_movies, num_features) = X.shape
    num_users = Theta.shape[0]
    for movie_idx in range(num_movies):
        query = {"movie_idx": movie_idx}
        update_field = {"x": list(X[movie_idx])}
        movie_collection.update(query, {"$set": update_field}, True)

    print "{0} movies in database.".format(movie_collection.count())

    for user_idx in range(num_users):
        query = {"user_idx": user_idx}
        update_field = {"x": list(Theta[user_idx])}
        user_collection.update(query, {"$set": update_field}, True)

    print "{0} users in database.".format(user_collection.count())