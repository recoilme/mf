// Simplified Collaborative Filtering using Matrix Factorization
// Ported from Python to Golang:
// https://medium.com/sfu-cspmp/recommendation-systems-collaborative-filtering-using-matrix-factorization-simplified-2118f4ef2cd3
// https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
package mf

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
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
						if usrF.At(i, k) != 1.0 {
							usrF.Set(i, k, usrF.At(i, k)+alpha*(2*eij*itemF.At(k, j)-beta*usrF.At(i, k)))
						}
						if itemF.At(k, j) != 1.0 {
							itemF.Set(k, j, itemF.At(k, j)+alpha*(2*eij*usrF.At(i, k)-beta*itemF.At(k, j)))
						}
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
func RatingLoad(r io.Reader, ratIdx, usrIdx, itmIdx int) (rating *mat.Dense, usrs, itms []string, err error) {
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
		//fmt.Printf("'%+v'\n", s)
		dim := strings.Split(s, "\t")
		//scan to sd
		sd := SparseData{}
		for i, val := range dim {
			switch i {
			case ratIdx:
				rat, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return rating, usrs, itms, err
				}
				sd.Rating = rat
			case usrIdx:
				if val == "" {
					return rating, usrs, itms, errors.New("empty user in line:" + s)
				}
				sd.User = val
				if !usrsMap[sd.User] {
					usrsMap[sd.User] = true
					usrs = append(usrs, sd.User)
				}
			case itmIdx:
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
			//fmt.Println(sd)
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
	//fmt.Println(rating)
	return rating, usrs, itms, nil
}

// RatingLoadCsv - convert csv matrix to tsv
// no error checks here
func RatingLoadCsv(r io.Reader) string {
	scanner := bufio.NewScanner(r)
	isHeader := true
	itms := make([]string, 0)
	usrs := make([]string, 0)
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	for scanner.Scan() {
		s := scanner.Text()
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		spl := strings.Split(s, ",")
		if isHeader {
			isHeader = false
			itms = append(itms, spl[1:]...)
			continue
			//fmt.Println(itms)
		}
		//fmt.Printf("'%+v'\n", s)
		for i, val := range spl {
			if i == 0 {
				usrs = append(usrs, spl[0])
				continue
			}
			w.WriteString(fmt.Sprintf("%s\t%s\t%s\n", val, spl[0], itms[i-1]))
		}
		//dim := strings.Split(s, "\t")
	}
	w.Flush()
	tsv := string(buf.Bytes())
	//fmt.Println(tsv)
	return tsv
}

func ItemLoad(ir, ur io.Reader, usrs, itms []string, rating *mat.Dense) ([]string, *mat.Dense, *mat.Dense) {

	//load items ft
	scanner := bufio.NewScanner(ir)

	ftMap := make(map[string]bool)
	fts := make([]string, 0)
	itemsFtMap := make(map[int]string)
	for scanner.Scan() {
		s := scanner.Text()
		if s == "" {
			continue
		}
		//fmt.Printf("'%+v'\n", s)
		dim := strings.Split(s, "\t")
		//scan to sd
		itmId := len(itms)
		for i, val := range dim {
			switch i {
			case 0:
				itmId = sort.SearchStrings(itms, val)
			default:
				// если есть такой итем запоминаем фичу
				//key:= fmt.Sprintf("%d:%s",itmId,)
				if itmId < len(itms) {
					if !ftMap[val] {
						ftMap[val] = true
						fts = append(fts, val)
					}
					itemsFtMap[itmId] = val
					//fmt.Println(itmId, val)
				}
			}
			//fmt.Println(sd)
		}
	}

	// load user ft
	scanner = bufio.NewScanner(ur)

	usrsFtMap := make(map[int]string)
	for scanner.Scan() {
		s := scanner.Text()
		if s == "" {
			continue
		}
		//fmt.Printf("'%+v'\n", s)
		dim := strings.Split(s, "\t")

		usrId := len(usrs)
		for i, val := range dim {
			switch i {
			case 0:
				usrId = sort.SearchStrings(usrs, val)
			default:
				// если есть такой user запоминаем фичу
				if usrId < len(usrs) {
					if !ftMap[val] {
						ftMap[val] = true
						fts = append(fts, val)
					}
					usrsFtMap[usrId] = val
					//fmt.Println(itmId, val)
				}
			}
			//fmt.Println(sd)
		}
	}

	sort.Sort(sort.StringSlice(fts))
	//fmt.Println(fts)
	itemFT := mat.NewDense(len(itms), len(fts), nil)
	//запишем фичи итема как 1 если есть
	for i := 0; i < len(itms); i++ {
		for j := 0; j < len(fts); j++ {
			if ft, ok := itemsFtMap[i]; ok {
				//если есть у итема фича пишем 1
				if ft == fts[j] {
					//fmt.Println(i, j, ft)
					itemFT.Set(i, j, 1)
				}
			}
		}
	}

	userF := mat.NewDense(len(usrs), len(fts), nil)
	//запишем фичи юзера как 1 если есть
	for i := 0; i < len(usrs); i++ {
		for j := 0; j < len(fts); j++ {
			if ft, ok := usrsFtMap[i]; ok {
				//если есть у юзера фича пишем 1
				if ft == fts[j] {
					//fmt.Println(i, j, ft)
					userF.Set(i, j, 1)
				}
			}
		}
	}
	/*
		//здесь бежим по матрице рейтинга
		for i := 0; i < len(usrs); i++ {
			for j := 0; j < len(itms); j++ {
				//пройдемся по фичам и найдем есть ли она у итема
				for ftId, val := range fts {
					// items
					if ft, ok := itemsFtMap[j]; ok && ft == val {
						//у этого итема есть фича ft, индекс фичи:
						//if ftId := sort.SearchStrings(fts, ft); ftId < len(fts) {
						// запишем рейтинг в матрицу фичей юзера
						userF.Set(i, ftId, rating.At(i, j))
						//}
					}
					// users
					if ft, ok := usrsFtMap[i]; ok && ft == val {
						//у этого юзера есть фича ft, индекс фичи:
						//if ftId := sort.SearchStrings(fts, ft); ftId < len(fts) {
						// запишем рейтинг в матрицу фичей итема
						itemFT.Set(j, ftId, rating.At(i, j))
						//}
					}
				}
				//пройдемся по фичам и найдем есть ли она у юзера
				for _, val := range fts {
					if ft, ok := usrsFtMap[i]; ok && ft == val {
						//у этого юзера есть фича ft, индекс фичи:
						if ftId := sort.SearchStrings(fts, ft); ftId < len(fts) {
							// запишем рейтинг в матрицу фичей итема
							itemFT.Set(j, ftId, rating.At(i, j))
						}
					}
				}
			}
		}
	*/
	return fts, itemFT, userF
}
