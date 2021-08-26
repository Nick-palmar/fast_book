## Collaborative Filtering Deep Dive

1. Collaborative filtering is useful for recommeding which item is likely to be useful for which user given a number of users and items in previous data. 

2. Collaborative filtering solves the problem in question 1 by finding other users who have similar liked items as the previous user and recommending other, new items that other users have used or liked. Furthermore, the model is based off of certain latent factors (embeddings) which are like unidentified features for the model to learn from. These latent facotrs get updated by user/item in order to predict whether or not a user will like a product. 

3. A collaborative filtering model may fail to be a useful recommendation system if there is bias in the data. For example, if the data contains representation bias in a movie database, where anime lovers are constantly rating movies very highly, then it may set the direction of the recommendations towards anime movies. As such, this can attract more anime lovers and create a positive feedback loop. For non-anime watchers, this can put the recommendation system out of balance and make it relatively useless simply due to the misrepresentation of the user base. Also, a recommendation system can fail to be useful if a new user/product is added and not eased into the system correctly. The latent factors for new users/items have not been tuned making it possible for bad recommendations if new users/products are not introduced well into the system. 

4. A cross tab representation of collaborative filtering puts users on one axis, items on the other axis, and rating at the corresponding location in the matrix for user/item. As such, every user-item relationship in terms of rating in shown and the 'blank' spaces are the places where we want to predict a rating using collaborative filtering. 

5. Code to create a cross tab of movie lens:
```
pivot_ratings = ratings.pivot(index='user', columns='movie', values='rating')
pivot_ratings['not_nan'] = pivot_ratings.count(axis=1)
pivot_ratings.sort_values(by='not_nan', ascending=False)
```

6. In general, the word 'latent' means hidden. In collabroative filtering, a latent factor is a hidden factor that can be thought of as a weighted feature applied to each user and item. We do not specifically describe these features, however, using SGD, the learner is able to learn about the correct combinations of these latent factors for users and for items thus uncovering hidden features in the data (more implicit and based on user ratings, not explicit). 

7. A dot product is a mathematical vector operation which does element wise multiplicaiton of vectors and adds the result. It can be thought of as a directional multiplicaiton of vectors (how much are two vectors going in the same direction). Dot product with python lists:
```
a = [1, 2, 3]
b = [4, 5, 6]
dot_prod = 0
for i in range(len(a)):
    dot_prod += a[i] * b[i]

print(dot_prod)
```
Or, simpler:
```
a = [1, 2, 3]
b = [4, 5, 6]
dot_prod = sum([a[i]*b[i] for i in range(len(a))])
dot_prod
```

8. .merge() in pandas allows for similar functinality to join in SQL. It will join two data frames on a certain column. 

9. An embedding matrix is a matrix the contains all the latent features for either users or items. When searching for a specific set of latent features, the embedding matrix is the matrix multiplied by the one hot encoded vector to provide the result (or in the case of pytorch, the embedding matrix is simply indexed into but the gradient is taken as if matrix multiplication was being done between the enmbedding matrix and a one hot encoded vector). 

10. An embedding is the result of indexing into an embedding matrix and taking the gradient as if the embedding index was calculated using matrix multiplication of the embedding matrix by one hot encoded vectors. Certain embedding corresponds to certain one hot encoded vector to get the correct values; it the embedding matrix is multiplied by the corresponding identity matrix (matrix of all one hot encoded vectors), then all embeddings are retured (identical to the embeding matrix). 

11. We need embedding even if we could use one-hot encoded vectors for the same thing because embedding allows for direct indexing into the embedding matrix which is much more efficient (both time and space) than doing a full matrix multiplicaiton. 

12. Before training, an embedding contains a number of randomly initialized weights (the specific number is given by n_factors) which will be optimized to become the optimzed latent features for users or items. 

13. Create a class   
```
class Car:
    def __init__(self, speed):
        self.speed = speed
    
    def check_speed(self):
        return self.speed
    
    def increase_speed(self, speed=10):
        self.speed += speed
        return self.speed

car = Car(100)
print(car.check_speed())
print(car.increase_speed())
print(car.increase_speed(20))
```

14. Assuming that x is the parameter in the forward method of a pytorch module subclass for collaborative filtering, x[:,0] will return the entire first column of x, which likely contains all the users passed in a minibatch to the forward method. 

15. Create dot product class
```
class DotProduct(Module):
  def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
    self.users = Embedding(n_users, n_factors)
    self.movies = Embedding(n_movies, n_factors)
    self.y_range = y_range
  
  def forward(self, x):
    # get all required users and movie embeddings
    users = self.users(x[:, 0])
    movies = self.movies(x[:, 1])
    # calculate the dot product, summing over the columns to provide a single value for each row/user
    return sigmoid_range((users*movies).sum(axis=1), *self.y_range)


model = DotProduct(n_users, n_movies, n_factors)
learn = Learner(dls, model, loss_func=MSELossFlat(), wd=0.1)
learn.fit_one_cycle(3, 1e-3)
```

16. A good loss function to use on MovieLens is MSE because it is a reasonable way to represent accuracy of predictions (can to square root to get actual distance value) given that this is a regression problem. 

17. If we were to use cross-entropy loss instead without changing the model, we would get an error because the predictions from our model are a single value, however, cross entropy loss requires multiple values. In order to use cross entropy loss, we would need to adapt the model to give us a prediction for each of the movie ratings (from 0 to 5 going up in increments of 0.5) so that we would have 11 outputs. Then we would apply softmax, followed by log, followed by nll to get the loss for our specific target. The NN form of collaborative filtering could adapt to this method. 

18. The use of bias in the dot product model is to account for tendencies/bias in user ratings (a certain user tends to rate movies very highly) or tendencies for specific movies (a certain movie tends to be rated very highly by all users) which could otherwise not be done by simply using the weights/embeddings in the model. 

19. Another name for weight decay (as a very similar idea) is L2 regularization. 

20. Weight decay equation:
```
wd_loss = loss + wd * sum(parameters**2)
```
This is because wd is adding sum of squares of parameters to the loss to make smaller weights (avoid steep peaks/valleys in the model). 

21. Gradient of weight decay:
```
params.grad += wd * (2 * params)
```
It helps to reduce the weights by limiting the capacity of the model. Large weights will be penalized by increasing the loss so the model will tend to make them smaller. Intuitively, certain weights will become small enough that they can almost be 'ignored', thus reducing model capacity without trading out model complexity (depth or number of weights). We can then train longer without overfitting with this reduced capacity model. 

22. Reducing weights leads to better generalizations because large weight values can lead to very steep peaks and valleys in the model. This means that slight changes in value can lead to drastically different predictions (overfitting). By having more rounded peaks and valley (caused by reducing the weights), the model is able to generalize better during inference and train for more epochs without overfitting on the training data. 

23. argsort in pytorch returns the indicies to sort a tensor along in ascedning manner given a dimension (axis for better term) to sort on.

24. 