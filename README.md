# mf - matrix factorization. 

## Simplified Collaborative Filtering using Matrix Factorization

Ported from Python to Golang:

https://medium.com/sfu-cspmp/recommendation-systems-collaborative-filtering-using-matrix-factorization-simplified-2118f4ef2cd3

https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b

## Usage

```
// MatrixFact - Simplified Collaborative Filtering using Matrix Factorization
// rating: rating matrix, row - user, col - item
// usrF: |U| * fCnt (User features matrix)
// itemF(Transposed):|I| * fCnt (Item features matrix)
// K: latent features
// steps: iterations, 5000
// alpha: learning rate, 0.0002
// beta: regularization parameter, 0.02
// return user features matrix and item features matrix (transposed)

	// 6: num of User
	// 4: num of Movie
	// rating 1-5 (0 - unknown)
	rating := mat.NewDense(6, 4, []float64{
		5, 3, 0, 1,
		4, 0, 0, 1,
		1, 1, 0, 5,
		1, 0, 0, 4,
		0, 1, 5, 4,
		2, 1, 3, 0,
	})
	// Num of Features
	cntF := 3
	users, items := rating.Dims()
	// some random
	userF := randMat(users, cntF)
	itemFT := randMat(items, cntF)

	userF, itemF := mf.MatrixFact(rating, userF, itemFT, cntF, 0, 0, 0)

	predictMat.Mul(userF, itemF)
	printm(&predictMat)

	// Input
	// 5, 3, 0, 1
	// Predict
	// 5.016017        2.898316        4.840158        0.999967
```

## Matrix Factorization

Since “Netflix Price Challenge”, Matrix Factorization has been one of the most famous and widely used Collaborative Filtering technique. So, what is factorization? In simpler terms, Factorization is the method of expressing something big as a product of smaller factors.


Similarly, Matrix Factorization finds two rectangular matrices with smaller dimensions to represent a big rating matrix (RM). These factors retain the dependencies and properties of the rating matrix. One matrix can be seen as the user matrix (UM) where rows represent users and columns are k latent factors. The other matrix is the item matrix (IM)where rows are k latent factors and columns represent items. Here k < number of items and k < number of users.


The latent factors are otherwise called as features. Features are characteristics of an item. In our case, features of the movies can be its genre, actors, plot, etc. The features chosen by the Matrix Factorization is irrelevant to the solution. Now the rating matrix is represented as a dot product of user and item matrix.
We’ll now try to understand the features and the dot product of the factor matrices using the figure below. We will ignore other ratings in the rating matrix image for simplification. We’ll assume out factored matrices have only two features F1 and F2. Having Ryan Reynolds in mind, we will assume F1 to be “If its a movie with Marvel characters?” and F2 to be “ If Ryan is there in the movie?”.


User Matrix: According to Ryan, if its a Marvel movie, he’ll give it 3 points and if he is in the movie, he’ll give it 2 more points (Typical Ryan!). 


Item Matrix: Item Matrix contains binary values where the value is 1 if conditions of features mentioned above are satisfied and 0 otherwise. By performing dot product of the user matrix and item matrix, Infinity War gets 3 and Deadpool gets a 5.

## Machine Learning

Machine Learning is the solution. The model learns to find latent factors to factorize the rating matrix. To arrive at the best approximation of the factors, RMSE(root mean squared error) is the cost function to be minimized. After Matrix factorization is done, squared error is calculated for every movie rating in the rating matrix and the root value of the mean of squared error values are minimized. In order to minimize RMSE to learn the factors we use Gradient Descent.