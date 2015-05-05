# Film-Recommender
Designed to recommend people right movies based on their previous ratings

Installation
------------

Make sure you have Python 2.7 (and the pip package manager).

You also need to install the `numpy` and `scipy` libraries by running in the command window

```
pip install numpy
pip install scipy
```

Use
---

To get the collaborative filtering algorithm to run by

```
python recoEngineTest.py
```
Make sure to uncomment the three load__intoDb function in the main function.

Then run the web_ML script by
```
python web_ML.py
```

When the server is on, go to http://localhost:5000/Movies/ and you will see the website that shows 5 random movie from the database for you to rate. There are 4 websites:

(1) http://localhost:5000/Movies/ shows initial movie to rate. After each rating, you will jump to (2)

(2) http://localhost:5000/Movies/updated showing the ratings you give to each movie. If one movie is not rated, it will still show a blank form

(3) http://localhost:5000/Movies/i  where i goes from 0-4. Shows the details of each movie. You can access this page from (1) and (2)

(4) http://localhost:5000/Movies/allmovies shows the list of all the movie in the database.

