// Simplified Collaborative Filtering using Matrix Factorization
// Ported from Python to Golang:
// https://medium.com/sfu-cspmp/recommendation-systems-collaborative-filtering-using-matrix-factorization-simplified-2118f4ef2cd3
// https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
package mf

import (
	"bufio"
	"errors"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

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
						usrF.Set(i, k, usrF.At(i, k)+alpha*(2*eij*itemF.At(k, j)-beta*usrF.At(i, k)))
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

// RatingLoad - read sparse tsv to rating matrix
// tsv format:rating\tuser\titem\n
// return rating matrix and ordered usrs/itms slice (for userId - User)
func RatingLoad(r io.Reader) (rating *mat.Dense, usrs, itms []string, err error) {
	type SparseData struct {
		User   string
		Item   string
		Rating float64
	}
	data := make([]SparseData, 0)
	scanner := bufio.NewScanner(r)

	usrsMap := make(map[string]bool)
	itemsMap := make(map[string]bool)
	for scanner.Scan() {
		s := scanner.Text()
		if s == "" {
			continue
		}
		//fmt.Printf("'%+v'\n", t)
		dim := strings.Split(s, "\t")
		//scan to sd
		sd := SparseData{}
		for i, val := range dim {
			switch i {
			case 0:
				rat, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return rating, usrs, itms, err
				}
				sd.Rating = rat
			case 1:
				if val == "" {
					return rating, usrs, itms, errors.New("empty user in line:" + s)
				}
				sd.User = val
				if !usrsMap[sd.User] {
					usrsMap[sd.User] = true
					usrs = append(usrs, sd.User)
				}
			case 2:
				if val == "" {
					return rating, usrs, itms, errors.New("empty item in line:" + s)
				}
				sd.Item = val
				if !itemsMap[sd.Item] {
					itemsMap[sd.Item] = true
					itms = append(itms, sd.Item)
				}
			default:
				continue
			}
		}
		data = append(data, sd)
	}
	// convert users/items to ordered slice
	sort.Sort(sort.StringSlice(usrs))
	sort.Sort(sort.StringSlice(itms))

	rating = mat.NewDense(len(usrs), len(itms), nil)
	for _, sd := range data {
		if usrId := sort.SearchStrings(usrs, sd.User); usrId < len(usrs) {
			if itmId := sort.SearchStrings(itms, sd.Item); itmId < len(itms) {
				rating.Set(usrId, itmId, sd.Rating)
			}
		}
	}

	return rating, usrs, itms, nil
}
