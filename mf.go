// Simplified Collaborative Filtering using Matrix Factorization
// Ported from Python to Golang:
// https://medium.com/sfu-cspmp/recommendation-systems-collaborative-filtering-using-matrix-factorization-simplified-2118f4ef2cd3
// https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
package mf

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// MatrixFact - Simplified Collaborative Filtering using Matrix Factorization
// rating: rating matrix, row - user, col - item
// usrF: |U| * fCnt (User features matrix)
// itemF(Transposed):|I| * fCnt (Item features matrix)
// K: latent features
// steps: iterations, 5000
// alpha: learning rate, 0.0002
// beta: regularization parameter, 0.02
// return user features matrix and item features matrix (transposed)
func MatrixFact(rating, usrF, itemFT *mat.Dense, fCnt, steps int, alpha, beta float64) (*mat.Dense, *mat.Dense) {
	itemF := mat.DenseCopyOf(itemFT.T()) //transpose item features
	if steps == 0 {
		steps = 5000
	}
	if alpha == 0 {
		alpha = 0.0002
	}
	if beta == 0 {
		beta = 0.02
	}

	rrow, rcol := rating.Dims()
	for s := 0; s < steps; s++ {
		for i := 0; i < rrow; i++ {
			for j := 0; j < rcol; j++ {
				if rating.At(i, j) > 0 {
					// calculate error
					eij := rating.At(i, j) - mat.Dot(usrF.RowView(i), itemF.ColView(j))

					for k := 0; k < fCnt; k++ {
						// calculate gradient with alpha and beta parameter
						//P[i][k] = P[i][k] + alpha*(2*eij*Q[k][j]-beta*P[i][k])
						usrF.Set(i, k, usrF.At(i, k)+alpha*(2*eij*itemF.At(k, j)-beta*usrF.At(i, k)))
						//Q[k][j] = Q[k][j] + alpha*(2*eij*P[i][k]-beta*Q[k][j])
						itemF.Set(k, j, itemF.At(k, j)+alpha*(2*eij*usrF.At(i, k)-beta*itemF.At(k, j)))
					}

				}
			}
		}
		var eR mat.Dense
		eR.Mul(usrF, itemF)
		e := float64(0)
		for i := 0; i < rrow; i++ {
			for j := 0; j < rcol; j++ {
				if rating.At(i, j) > 0 {
					//e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
					e = e + math.Pow(rating.At(i, j)-mat.Dot(usrF.RowView(i), itemF.ColView(j)), 2.0)
					for k := 0; k < fCnt; k++ {
						e = e + (beta/2.)*(math.Pow(usrF.At(i, k), 2)+math.Pow(itemF.At(k, j), 2))
					}
				}
			}
		}
		// 0.001: local minimum
		if e < 0.001 {
			break
		}
	}
	return usrF, itemF
}
