import numpy as np
from flask import Flask, url_for, request, redirect, render_template
from pymongo import MongoClient
import recoEngineTest
import random
import algorithm

#Parameter Settings:
num=5   #number of movies shown to user
num_features=10


#Setting up database:
client = MongoClient('localhost', 27017)
#client.drop_database('mlMovieRating')
movie=client.mlMovieDb.movie        #trained movie data and features
dbRating = client.mlMovieRating
movie_to_rate = dbRating.movie_to_rate  #user's rating will be save into this collection


#initiating global variables:
num_movie=movie.count()
Movie_chosen_idx=[]
Movie_chosen=[]
Movie_chosen_X=[]
for i in range(num):
    adding=int(num_movie*random.random())
    Movie_chosen_idx.append(adding)
    Movie_chosen.append(movie.find()[i])
    Movie_chosen_X.append(movie.find()[i]['x'])
nametag=[]
for i in range(num):
    nametag.append('Name'+str(i))
app = Flask(__name__)


def prepareDB():
    recoEngineTest.loadMoviesIntoDBFromFileTest()
    recoEngineTest.loadUsersIntoDBFromFileTest()
    #recoEngineTest.loadRatingsIntoDBFromFileTest()

@app.route('/Movies/', methods=['GET','POST'])
def ML():
    if request.method == 'POST':
        for i in range(num):
            if nametag[i] in request.form:
                rating = request.form[nametag[i]]
                newRating={'movie_idx':Movie_chosen_idx[i],'Name':Movie_chosen[i]['movie_name'],'Rating':rating,'x':Movie_chosen_X[i]}
                movie_to_rate.delete_many({'Name':Movie_chosen[i]['movie_name']})
                movie_to_rate.insert(newRating)
        return render_template('ML_updated.html',Movie_chosen=Movie_chosen,num=num,movie_to_rate=movie_to_rate,nametag=nametag)
    else:
        return render_template('ML.html',Movie_chosen=Movie_chosen,nametag=nametag,num=num)


@app.route('/')
@app.route('/Movies/updated/', methods=['GET','POST'])
def ML_updated():
    if request.method == 'POST':
        for i in range(num):
            if nametag[i] in request.form:
                rating = request.form[nametag[i]]
                newRating={'movie_idx':Movie_chosen_idx[i],'Name':Movie_chosen[i]['movie_name'],'Rating':rating,'x':Movie_chosen_X[i]}
                movie_to_rate.delete_many({'Name':Movie_chosen[i]['movie_name']})
                movie_to_rate.insert(newRating)
    return render_template('ML_updated.html',Movie_chosen=Movie_chosen,num=num,movie_to_rate=movie_to_rate,nametag=nametag)


@app.route('/Movies/<int:movie_id>/', methods=['GET','POST'])
def ML_newMovie(movie_id):
    if request.method == 'POST':
        rating = request.form['name']
        newRating={'movie_idx':Movie_chosen_idx[movie_id],'Name':Movie_chosen[movie_id]['movie_name'],'Rating':rating,'x':Movie_chosen_X[movie_id]}
        movie_to_rate.delete_many({'Name':Movie_chosen[movie_id]['movie_name']})
        movie_to_rate.insert(newRating)
        return redirect(url_for('ML_updated',Movie_chosen=Movie_chosen,num=num,movie_to_rate=movie_to_rate,nametag=nametag))
    else:
        return render_template('ML_newMovie.html',Movie_chosen=Movie_chosen,movie_id=movie_id)


@app.route('/Movies/reRate/')
def ML_reRate():
    movie_to_rate.remove({})
    return redirect(url_for('ML_updated'))

@app.route('/Movies/allMovies/')
def ML_allMovies():
    movie=client.mlMovieDb.movie
    num_movie=movie.count()
    return render_template('ML_allMovies.html',movie=movie,num_movie=num_movie)

def optimizeTheta():
    num_rated=movie_to_rate.count()
    Y=np.zeros((num_rated,1))
    X=np.zeros((num_rated,num_features))
    for i in range(num_rated):
        Y[i]=int(movie_to_rate.find()[i]['Rating'])
        X[i]=movie_to_rate.find()[i]['x']
    R=np.ones((num_rated,1))
    Reg=1   #Regulization
    LearnRate=0.001
    IterationTimes=1000
    theta_init=np.random.rand(1,num_features+1)
    (Theta_list,Error_list)=algorithm.runIterate(theta_init,X,R,Y,Reg,LearnRate,IterationTimes)

    k=0
    Error_min=Error_list[k]
    for i in range(len(Error_list)):
        if Error_list[i]<Error_min:
            k=i
            Error_min=Error_list[k]
    print "Error_list:"+str(Error_list)
    print "Minimum Error="+str(Error_min)
    print "Optimal preference="+str(Theta_list[k])
    print "Best guess of rating="+str((np.dot(X,Theta_list[k].transpose()[1:,:])+Theta_list[k][0,0]).transpose())
    print "User rating:"+str(Y.transpose())

    return (Theta_list[k],Error_list[k],Theta_list, Error_list)


def recommendMovies(Theta):
    X_all=np.zeros((num_movie,num_features))
    for i in range(num_movie):
        X_all[i]=movie.find()[i]['x']
    theta=np.matrix(Theta).transpose()
    #print theta
    #print X_all
    rating_calculate=np.dot(X_all,theta[1:])+theta[0]
    #print rating_calculate
    index=np.matrix(np.linspace(1,num_movie,num_movie)).transpose()
    rating_with_index=np.zeros((num_movie,2))
    rating_with_index[:,:1]=index
    rating_with_index[:,1:]=rating_calculate
    rating_sort=rating_with_index[np.array(rating_with_index[:,1].argsort(axis=0).tolist()).ravel()]
    print rating_with_index
    print rating_sort

    return (rating_sort,rating_with_index)


if __name__ == '__main__':
    #prepareDB()
    (Theta,Error,Theta_list, Error_list)=optimizeTheta()
    recommendMovies(Theta)
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

    #loadRatingsIntoDBFromFileTest()
    client.close()