package mf_test

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/recoilme/mf"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func Test_Base(t *testing.T) {
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
	// Num of Features
	cntF := 3
	users, items := rating.Dims()
	userF := randMat(users, cntF)
	itemFT := randMat(items, cntF)

	userF, itemF := mf.MatrixFact(rating, userF, itemFT, cntF, 0, 0, 0)

	println("userF")
	printm(userF)
	println("itemF")
	printm(itemF)
	var predictMat mat.Dense
	println("predictMat")
	predictMat.Mul(userF, itemF)
	printm(&predictMat)
	/*  Python
		M1			M2			M3			M4
	U1	4.9921822	2.94665346	3.40536863	1.00099803
	U2	3.97665318	2.15037496	3.10801712	0.99668824
	U3	1.04690221	0.89961819	5.20187891	4.96401721
	U4	0.98735764	0.77053916	4.27454201	3.97482436
	U5	1.86869693	1.09103752	4.94375052	4.02158161
	U6	1.93781997	1.06363722	3.01008954	1.99265014
	Go
	predictMat
	5.016017        2.898316        4.840158        0.999967
	3.967485        2.261877        4.110961        0.996784
	1.083118        0.811969        4.775191        4.965624
	0.975709        0.667362        4.048701        3.973770
	1.931771        1.198243        4.913493        4.026525
	1.895282        1.108765        3.021853        1.738874
	Input
	5, 3, 0, 1
	Predict
	5.016017        2.898316        4.840158        0.999967
	*/
}

func printm(m *mat.Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%f\t", m.At(i, j))
		}
		println("")
	}
}

func randMat(row, col int) *mat.Dense {
	r := mat.NewDense(row, col, nil)
	for i := 0; i < row; i++ {
		for j := 0; j < col; j++ {
			r.Set(i, j, rand.Float64())
		}
	}
	return r
}

func Test_Read(t *testing.T) {
	data :=
		`
1	u1	i1	f1
0	u1	i2	f2
`
	_ = data
	sd, err := linesRead(strings.NewReader(data))
	assert.NoError(t, err)
	fmt.Printf("%+v\n", sd)
}

type SparseData struct {
	User   string
	Item   string
	Ft     []string
	Rating float64
}

func linesRead(r io.Reader) ([]SparseData, error) {
	data := make([]SparseData, 0)
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		s := scanner.Text()
		if s == "" {
			continue
		}
		//fmt.Printf("'%+v'\n", t)
		dim := strings.Split(s, "\t")
		//scan to sd
		sd := SparseData{}
		ft := make([]string, 0)
		for i, val := range dim {
			switch i {
			case 0:
				rat, err := strconv.ParseFloat(val, 64)
				if err != nil {
					return nil, err
				}
				sd.Rating = rat
			case 1:
				sd.User = val
			case 2:
				sd.Item = val
			default:
				ft = append(ft, val)
			}
		}
		sd.Ft = ft
		data = append(data, sd)
	}
	return data, nil
}
