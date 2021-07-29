package mf_test

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
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
	cntF := 2
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

func Test_LoadRating(t *testing.T) {
	data :=
		`
1	u1	i1
0	u1	i2
1	u1	i3
0	u2	i4		
`
	r := strings.NewReader(data)
	rating, usrs, itms, err := mf.RatingLoad(r, 0, 1, 2)

	assert.NoError(t, err)
	assert.Equal(t, 2, len(usrs))
	assert.Equal(t, 4, len(itms))
	usrId := sort.SearchStrings(usrs, "u1")
	assert.Equal(t, 0, usrId)
	itmId := sort.SearchStrings(itms, "i3")
	//fmt.Println(itms)
	assert.Equal(t, 2, itmId)
	rat := rating.At(usrId, itmId)
	assert.Equal(t, float64(1), rat)
}

func Test_LoadCsv(t *testing.T) {
	data :=
		`
		row_id,Гарри_Поттер,Хоббит_Нежданное_путешествие,Хоббит_Пустошь_Смауга,Хроники_Нарнии_Принц_Каспиан,Сердце_дракона,Аниме
		a,1,1,1,0,0,0
		b,1,1,1,0,0,0
		c,1,0,0,1,0,0
		d,1,0,1,0,0,0
		e,1,1,0,0,1,0
		f,0,0,0,0,0,1
`
	tsv := mf.RatingLoadCsv(strings.NewReader(data))
	r := strings.NewReader(tsv)
	rating, usrs, itms, err := mf.RatingLoad(r, 0, 1, 2)
	assert.NoError(t, err)
	assert.Equal(t, 6, len(usrs))
	assert.Equal(t, 6, len(itms))
	_ = rating
	//printm(rating)
}

func Test_LoadTsv(t *testing.T) {
	f, err := os.Open("testdata/rating.tsv")
	assert.NoError(t, err)
	defer f.Close()

	rating, usrs, itms, err := mf.RatingLoad(f, 0, 1, 2)
	assert.NoError(t, err)
	_ = usrs
	_ = itms
	cntF := 4
	users, items := rating.Dims()
	userF := randMat(users, cntF)
	itemFT := randMat(items, cntF)
	//посмотрим что предиктнет гарри поттеру, стерев часть оценок
	rating.Set(0, 0, 0)
	rating.Set(1, 0, 0)
	rating.Set(3, 0, 0)
	userF, itemF := mf.MatrixFact(rating, userF, itemFT, cntF, 0, 0, 0)

	var predictMat mat.Dense
	println("predictMat")
	fmt.Println(itms)
	predictMat.Mul(userF, itemF)
	printm(&predictMat)
	//printm(rating)
}

func Test_LoadItem(t *testing.T) {
	f, err := os.Open("testdata/rating.tsv")
	assert.NoError(t, err)
	rating, usrs, itms, err := mf.RatingLoad(f, 0, 1, 2)
	assert.NoError(t, err)
	f.Close()

	fi, err := os.Open("testdata/item.tsv")
	assert.NoError(t, err)
	defer fi.Close()

	fu, err := os.Open("testdata/user.tsv")
	assert.NoError(t, err)
	defer fu.Close()

	fts, itemFT, userF := mf.ItemLoad(fi, fu, usrs, itms, rating)
	println("fts")
	fmt.Println(fts)
	println("userF")
	fmt.Println(usrs)
	printm(userF)
	println("itemF")
	fmt.Println(itms)
	printm(itemFT)

	//посчитаем
	userF, itemF := mf.MatrixFact(rating, userF, itemFT, len(fts), 0, 0, 0)

	println("userF after GD")
	fmt.Println(usrs)
	printm(userF)

	println("itemF after GD")
	fmt.Println(itms)
	itemFOrig := mat.DenseCopyOf(itemF.T()) //transpose item features
	printm(itemFOrig)

	var predictMat mat.Dense
	println("predictMat")
	predictMat.Mul(userF, itemF)
	printm(&predictMat)
}
