# mf - matrix factorization. Simplified Collaborative Filtering using Matrix Factorization

Ported from Python to Golang:

https://medium.com/sfu-cspmp/recommendation-systems-collaborative-filtering-using-matrix-factorization-simplified-2118f4ef2cd3

https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b

## Usage

// MatrixFact - Simplified Collaborative Filtering using Matrix Factorization
// rating: rating matrix, row - user, col - item
// usrF: |U| * fCnt (User features matrix)
// itemF(Transposed):|I| * fCnt (Item features matrix)
// K: latent features
// steps: iterations, 5000
// alpha: learning rate, 0.0002
// beta: regularization parameter, 0.02
// return user features matrix and item features matrix (transposed)

```
	rating := mat.NewDense(6, 4, []float64{
		5, 3, 0, 1,
		4, 0, 0, 1,
		1, 1, 0, 5,
		1, 0, 0, 4,
		0, 1, 5, 4,
		2, 1, 3, 0,
	})
	// 6: num of User
	// 4: num of Movie
    // rating 1-5 (0 - unknown)
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