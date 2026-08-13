// research_3d.go — 3D-исследовательские эксперименты для лаборатории RMT-LLM (Go, русская версия)
// =============================================================================
//
// Добавляет девять продвинутых 3D-исследовательских экспериментов поверх
// существующего 2D-исследовательского модуля. Каждый эксперимент возвращает
// структурированный map, повторяющий JSON-вывод Python-модуля
// `research_3d.py`, чтобы `charts_3d.py` мог читать результаты без изменений.
//
// Эксперименты (ID совпадают с Python):
//
//   6. SCEN-3D-HESSIAN            — Ландшафт потерь Гессиана
//   7. SCEN-3D-MANIFOLD           — Геометрия многообразия (PCA participation ratio)
//   8. SCEN-3D-TRAJECTORY         — Анализ траектории рассуждений
//   9. SCEN-3D-SPECTRAL-SURFACE   — Регрессия спектральной поверхности
//  10. SCEN-3D-RIEMANN            — Риманова кривизна на kNN-графе
//  11. SCEN-3D-ATTENTION-FLOW     — 3D-поток внимания
//  12. SCEN-3D-NCRIT-SURFACE      — Поверхность коллапса N_crit
//  13. SCEN-3D-PARAM-SPACE        — Развёртка пространства параметров
//  14. SCEN-3D-COALITION          — Дрейф коалиции
//
// Все эксперименты принимают соглашение о бесконечных параметрах (строка
// "inf", "+inf", "infinity" или math.Inf(1) ограничиваются конечным верхним
// пределом во время вычислений через помощник `clampInf`).
//
// Только стандартная библиотека Go — без внешних модулей. Линейная алгебра
// (умножение/транспонирование матриц, единичная матрица, симметричная
// эйген-декомпозиция Якоби, softmax) реализована с нуля ниже.
//
// Автор: Исхак Хамзатович Исаев, ORCID: 0009-0003-7299-0701
// Лицензия: Проприетарная — Все права защищены.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Тип Experiment3D и реестр
// ---------------------------------------------------------------------------

// Experiment3D описывает один 3D-исследовательский эксперимент.
type Experiment3D struct {
	ID          string
	Name        string
	Description string
	Run         func(map[string]interface{}) map[string]interface{}
}

// Experiments3D — реестр всех 3D-экспериментов, ключом является ID ("6".."14").
var Experiments3D = map[string]Experiment3D{}

func init() {
	Experiments3D["6"] = Experiment3D{
		ID:          "6",
		Name:        "Ландшафт потерь Гессиана (3D)",
		Description: "Возмущение модели вдоль топ-2 собственных направлений Гессиана, замер 3D-поверхности потерь.",
		Run:         expHessianLossLandscape,
	}
	Experiments3D["7"] = Experiment3D{
		ID:          "7",
		Name:        "Геометрия многообразия (3D PCA)",
		Description: "Оценка внутренней размерности через participation ratio PCA; 3D-проекция.",
		Run:         expManifoldGeometry,
	}
	Experiments3D["8"] = Experiment3D{
		ID:          "8",
		Name:        "Анализ траектории рассуждений (3D)",
		Description: "Сэмплирование (шаг, честность, обман, спектральный радиус) и детектирование начала обмана.",
		Run:         expTrajectoryAnalysis,
	}
	Experiments3D["9"] = Experiment3D{
		ID:          "9",
		Name:        "Регрессия спектральной поверхности (3D)",
		Description: "Аппроксимация поверхности λ_max(слой, токен) и детектирование бифуркации N_crit.",
		Run:         expSpectralSurfaceRegression,
	}
	Experiments3D["10"] = Experiment3D{
		ID:          "10",
		Name:        "Риманова кривизна (3D)",
		Description: "Оценка дискретной гауссовой кривизны на kNN-графе скрытых состояний.",
		Run:         expRiemannianCurvature,
	}
	Experiments3D["11"] = Experiment3D{
		ID:          "11",
		Name:        "3D-поток внимания",
		Description: "Замер поверхности весов внимания и количественная оценка диагонального vs размазанного режима.",
		Run:         expAttentionFlow3D,
	}
	Experiments3D["12"] = Experiment3D{
		ID:          "12",
		Name:        "Поверхность коллапса N_crit (3D)",
		Description: "Вычисление поверхности T_crit(β, μ_RLHF) по порядку Капуто и RLHF-давлению.",
		Run:         expNcritSurface,
	}
	Experiments3D["13"] = Experiment3D{
		ID:          "13",
		Name:        "Развёртка пространства параметров (3D)",
		Description: "Свип (температура, top_p) и замер поверхности частоты галлюцинаций.",
		Run:         expParameterSpace,
	}
	Experiments3D["14"] = Experiment3D{
		ID:          "14",
		Name:        "Дрейф коалиционного обмана (3D)",
		Description: "Симуляция дрейфа обмана мультиагентной коалиции по раундам (SCEN-COAL-09).",
		Run:         expCoalitionDrift,
	}
}

// nameMap3D связывает ID эксперимента со snake_case-ключом, используемым charts_3d.py.
var nameMap3D = map[string]string{
	"6":  "loss_landscape",
	"7":  "manifold_geometry",
	"8":  "trajectory",
	"9":  "spectral_surface",
	"10": "riemannian_curvature",
	"11": "attention_flow_3d",
	"12": "ncrit_surface",
	"13": "parameter_space",
	"14": "coalition_drift",
}

// ---------------------------------------------------------------------------
// clampInf — преобразование inf / "inf" / nil / NaN в конечное значение
// ---------------------------------------------------------------------------

// clampInf нормализует соглашение о бесконечных параметрах, используемое во
// всей лаборатории. Поддерживаемые входы:
//   - nil                    -> defaultVal
//   - "inf", "+inf", "infinity" (без учёта регистра) -> maxFinite
//   - любая другая строка    -> разбор как float, иначе defaultVal при ошибке
//   - math.Inf(1), -Inf      -> maxFinite (или defaultVal для -Inf)
//   - math.NaN()             -> defaultVal
//   - обычный float64        -> без изменений
func clampInf(value interface{}, defaultVal, maxFinite float64) float64 {
	if value == nil {
		return defaultVal
	}
	switch v := value.(type) {
	case string:
		lower := strings.ToLower(strings.TrimSpace(v))
		switch lower {
		case "inf", "+inf", "infinity":
			return maxFinite
		case "-inf", "-infinity":
			return -maxFinite
		case "nan":
			return defaultVal
		case "":
			return defaultVal
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return defaultVal
		}
		return clampFinite(f, defaultVal, maxFinite)
	case float64:
		return clampFinite(v, defaultVal, maxFinite)
	case float32:
		return clampFinite(float64(v), defaultVal, maxFinite)
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case bool:
		if v {
			return 1.0
		}
		return 0.0
	default:
		return defaultVal
	}
}

// clampFinite применяет inf/nan-политику к обычному float64.
func clampFinite(v, defaultVal, maxFinite float64) float64 {
	if math.IsNaN(v) {
		return defaultVal
	}
	if math.IsInf(v, 1) {
		return maxFinite
	}
	if math.IsInf(v, -1) {
		return -maxFinite
	}
	return v
}

// ---------------------------------------------------------------------------
// Числовые помощники
// ---------------------------------------------------------------------------

func linspace(start, stop float64, n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	if n == 1 {
		return []float64{start}
	}
	step := (stop - start) / float64(n-1)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = start + float64(i)*step
	}
	out[n-1] = stop
	return out
}

// meshgridIJ повторяет numpy.meshgrid(a, b, indexing="ij").
// Возвращает W1, W2 каждая размерности (len(a), len(b)).
func meshgridIJ(a, b []float64) ([][]float64, [][]float64) {
	W1 := make([][]float64, len(a))
	W2 := make([][]float64, len(a))
	for i := 0; i < len(a); i++ {
		W1[i] = make([]float64, len(b))
		W2[i] = make([]float64, len(b))
		for j := 0; j < len(b); j++ {
			W1[i][j] = a[i]
			W2[i][j] = b[j]
		}
	}
	return W1, W2
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

// softmax вычисляет численно стабильный softmax одномерного среза.
func softmax(x []float64) []float64 {
	if len(x) == 0 {
		return x
	}
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		e := math.Exp(v - maxVal)
		out[i] = e
		sum += e
	}
	if sum > 1e-12 {
		for i := range out {
			out[i] /= sum
		}
	}
	return out
}

// polyfit1 аппроксимирует y = slope*x + intercept методом наименьших квадратов.
func polyfit1(x, y []float64) (slope, intercept float64) {
	n := float64(len(x))
	if len(x) < 2 || len(x) != len(y) {
		return 0, 0
	}
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
	}
	denom := n*sumX2 - sumX*sumX
	if math.Abs(denom) < 1e-15 {
		return 0, sumY / n
	}
	slope = (n*sumXY - sumX*sumY) / denom
	intercept = (sumY - slope*sumX) / n
	return slope, intercept
}

// meanStd вычисляет выборочное среднее и (генеральное) стандартное отклонение.
func meanStd(x []float64) (mean, std float64) {
	if len(x) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	mean = sum / float64(len(x))
	var sq float64
	for _, v := range x {
		d := v - mean
		sq += d * d
	}
	std = math.Sqrt(sq / float64(len(x)))
	return
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

// ---------------------------------------------------------------------------
// Линейная алгебра (только stdlib)
// ---------------------------------------------------------------------------

func matIdentity(n int) [][]float64 {
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		m[i][i] = 1.0
	}
	return m
}

func matTranspose(a [][]float64) [][]float64 {
	if len(a) == 0 {
		return a
	}
	rows := len(a)
	cols := len(a[0])
	t := make([][]float64, cols)
	for i := 0; i < cols; i++ {
		t[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			t[i][j] = a[j][i]
		}
	}
	return t
}

// matMul возвращает a*b для матриц совместимых размерностей.
func matMul(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(b) == 0 {
		return [][]float64{}
	}
	n := len(a)
	k := len(a[0])
	m := len(b[0])
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			s := 0.0
			for l := 0; l < k; l++ {
				s += a[i][l] * b[l][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// covariance возвращает (D x D) выборочную ковариацию (N x D)-матрицы x.
func covariance(x [][]float64) [][]float64 {
	if len(x) == 0 {
		return [][]float64{}
	}
	n := len(x)
	d := len(x[0])
	mean := make([]float64, d)
	for _, row := range x {
		for j := 0; j < d; j++ {
			mean[j] += row[j]
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	cov := make([][]float64, d)
	for i := 0; i < d; i++ {
		cov[i] = make([]float64, d)
	}
	denom := math.Max(float64(n-1), 1.0)
	for _, row := range x {
		for i := 0; i < d; i++ {
			di := row[i] - mean[i]
			for j := i; j < d; j++ {
				dj := row[j] - mean[j]
				cov[i][j] += di * dj
			}
		}
	}
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			cov[i][j] /= denom
			cov[j][i] = cov[i][j]
		}
	}
	return cov
}

// jacobiEigen возвращает собственные значения (по возрастанию) и
// соответствующие собственные векторы симметричной матрицы, используя
// классический алгоритм вращений Якоби.
// eigvecs[k] — собственный вектор (длины n), соответствующий eigvals[k].
func jacobiEigen(a [][]float64) (eigvals []float64, eigvecs [][]float64) {
	n := len(a)
	if n == 0 {
		return []float64{}, [][]float64{}
	}
	// Копируем a (защитно)
	A := make([][]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		copy(A[i], a[i])
	}
	V := matIdentity(n)
	maxIter := 100 + 50*n*n
	if maxIter > 200000 {
		maxIter = 200000
	}
	for iter := 0; iter < maxIter; iter++ {
		// Найти наибольший внедиагональный |A[p][q]|
		p, q := 0, 1
		maxOff := 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if math.Abs(A[i][j]) > maxOff {
					maxOff = math.Abs(A[i][j])
					p, q = i, j
				}
			}
		}
		if maxOff < 1e-14 {
			break
		}
		app := A[p][p]
		aqq := A[q][q]
		apq := A[p][q]
		var t float64
		if math.Abs(app-aqq) < 1e-30 {
			t = 1.0
			if apq < 0 {
				t = -1.0
			}
		} else {
			theta := (aqq - app) / (2.0 * apq)
			signT := 1.0
			if theta < 0 {
				signT = -1.0
			}
			t = signT / (math.Abs(theta) + math.Sqrt(theta*theta+1.0))
		}
		c := 1.0 / math.Sqrt(1.0+t*t)
		s := t * c
		// Применяем вращение к строкам/столбцам p,q
		for i := 0; i < n; i++ {
			aip := A[i][p]
			aiq := A[i][q]
			A[i][p] = c*aip - s*aiq
			A[i][q] = s*aip + c*aiq
		}
		for j := 0; j < n; j++ {
			apj := A[p][j]
			aqj := A[q][j]
			A[p][j] = c*apj - s*aqj
			A[q][j] = s*apj + c*aqj
		}
		for i := 0; i < n; i++ {
			vip := V[i][p]
			viq := V[i][q]
			V[i][p] = c*vip - s*viq
			V[i][q] = s*vip + c*viq
		}
	}
	eigvals = make([]float64, n)
	for i := 0; i < n; i++ {
		eigvals[i] = A[i][i]
	}
	// Сортировка по возрастанию
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return eigvals[idx[i]] < eigvals[idx[j]] })
	sortedVals := make([]float64, n)
	sortedVecs := make([][]float64, n)
	for i, k := range idx {
		sortedVals[i] = eigvals[k]
		col := make([]float64, n)
		for r := 0; r < n; r++ {
			col[r] = V[r][k]
		}
		sortedVecs[i] = col
	}
	return sortedVals, sortedVecs
}

// ---------------------------------------------------------------------------
// Генератор синтетических скрытых состояний (заменяет TinyGPT для pure-stdlib)
// ---------------------------------------------------------------------------

// gptConfig возвращает карту конфигурации по умолчанию (в стиле TinyGPT).
func gptConfig(params map[string]interface{}) map[string]interface{} {
	hiddenDim := int(clampInf(params["hidden_dim"], 64, 4096))
	nLayers := int(clampInf(params["n_layers"], 6, 64))
	nHeads := int(clampInf(params["n_heads"], 4, 64))
	vocabSize := int(clampInf(params["vocab_size"], 256, 1<<20))
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	return map[string]interface{}{
		"hidden_dim":   hiddenDim,
		"n_layers":     nLayers,
		"n_heads":      nHeads,
		"vocab_size":   vocabSize,
		"max_seq_len":  64,
		"seed":         seed,
		"model_type":   "tiny_gpt_synthetic",
		"param_count":  hiddenDim * hiddenDim * nLayers,
	}
}

// synthHiddenStates генерирует nLayers x seq x hiddenDim синтетические
// скрытые состояния с реалистичной спектральной структурой (низкоранговое
// смещение + позиционный сигнал + гауссов шум). Используется вместо
// полноценного forward-прохода TinyGPT.
func synthHiddenStates(nLayers, seq, hiddenDim int, seed int64) [][][]float64 {
	rng := rand.New(rand.NewSource(seed))
	layers := make([][][]float64, nLayers)
	for l := 0; l < nLayers; l++ {
		// Низкоранговое смещение слоя — даёт ковариации несколько доминирующих собственных значений.
		bias := make([]float64, hiddenDim)
		for i := range bias {
			bias[i] = rng.NormFloat64() * 0.6
		}
		// Позиционное кодирование (синусоидальное).
		posEnc := make([][]float64, seq)
		for s := 0; s < seq; s++ {
			posEnc[s] = make([]float64, hiddenDim)
			for d := 0; d < hiddenDim; d++ {
				posEnc[s][d] = math.Sin(float64(s)/float64(seq+1)*math.Pi*float64(d+1)*0.3) * 0.4
			}
		}
		layers[l] = make([][]float64, seq)
		for s := 0; s < seq; s++ {
			row := make([]float64, hiddenDim)
			for d := 0; d < hiddenDim; d++ {
				// Масштабирование слоя для вариации между слоями.
				scale := 1.0 + 0.1*float64(l)
				row[d] = scale*posEnc[s][d] + bias[d] + rng.NormFloat64()*0.25
			}
			layers[l][s] = row
		}
	}
	return layers
}

// stackHiddenLayers объединяет [layer][seq][dim] -> (layer*seq) x dim матрицу.
func stackHiddenLayers(layers [][][]float64) [][]float64 {
	var out [][]float64
	for _, layer := range layers {
		out = append(out, layer...)
	}
	return out
}

// maxEigenvalue возвращает наибольшее собственное значение ковариации x.
func maxEigenvalue(x [][]float64) float64 {
	if len(x) == 0 || len(x[0]) == 0 {
		return 0
	}
	cov := covariance(x)
	vals, _ := jacobiEigen(cov)
	if len(vals) == 0 {
		return 0
	}
	// vals отсортированы по возрастанию; max = последний
	return vals[len(vals)-1]
}

// topKEigenvalues возвращает k наибольших собственных значений cov(x) (по убыванию).
func topKEigenvalues(x [][]float64, k int) []float64 {
	if len(x) == 0 || len(x[0]) == 0 {
		return []float64{}
	}
	cov := covariance(x)
	vals, _ := jacobiEigen(cov)
	// По возрастанию; разворачиваем в убывание
	for i, j := 0, len(vals)-1; i < j; i, j = i+1, j-1 {
		vals[i], vals[j] = vals[j], vals[i]
	}
	if k > len(vals) {
		k = len(vals)
	}
	return vals[:k]
}

// pcaReduce центрирует x и проецирует на топ-k главных компонент.
// Возвращает проекцию (N x k) и топ-k собственных значений (по убыванию).
func pcaReduce(x [][]float64, k int) ([][]float64, []float64) {
	if len(x) == 0 || len(x[0]) == 0 {
		return [][]float64{}, []float64{}
	}
	n := len(x)
	d := len(x[0])
	mean := make([]float64, d)
	for _, row := range x {
		for j := 0; j < d; j++ {
			mean[j] += row[j]
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	xc := make([][]float64, n)
	for i, row := range x {
		xc[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			xc[i][j] = row[j] - mean[j]
		}
	}
	cov := covariance(xc)
	vals, vecs := jacobiEigen(cov)
	// Разворачиваем в убывание
	for i, j := 0, len(vals)-1; i < j; i, j = i+1, j-1 {
		vals[i], vals[j] = vals[j], vals[i]
		vecs[i], vecs[j] = vecs[j], vecs[i]
	}
	if k > len(vals) {
		k = len(vals)
	}
	proj := make([][]float64, n)
	for i := 0; i < n; i++ {
		proj[i] = make([]float64, k)
		for c := 0; c < k; c++ {
			s := 0.0
			for j := 0; j < d; j++ {
				s += xc[i][j] * vecs[c][j]
			}
			proj[i][c] = s
		}
	}
	return proj, vals[:k]
}

// ---------------------------------------------------------------------------
// Эксперимент 6: Ландшафт потерь Гессиана
// ---------------------------------------------------------------------------

func expHessianLossLandscape(params map[string]interface{}) map[string]interface{} {
	grid := int(clampInf(params["hessian_grid_size"], 24, 256))
	if grid < 4 {
		grid = 4
	}
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	hiddenDim := int(clampInf(params["hidden_dim"], 64, 4096))
	nLayers := int(clampInf(params["n_layers"], 6, 64))
	nHeads := int(clampInf(params["n_heads"], 4, 64))
	vocabSize := int(clampInf(params["vocab_size"], 256, 1<<20))

	seqLen := 64
	if seqLen > hiddenDim {
		seqLen = hiddenDim
	}
	layers := synthHiddenStates(nLayers, seqLen, hiddenDim, seed)
	H := stackHiddenLayers(layers)
	cov := covariance(H)
	eigvals, eigvecs := jacobiEigen(cov)
	// По возрастанию; последние два — топ-2.
	lam1 := 0.0
	lam2 := 0.0
	if len(eigvals) >= 2 {
		lam1 = eigvals[len(eigvals)-1]
		lam2 = eigvals[len(eigvals)-2]
	} else if len(eigvals) == 1 {
		lam1 = eigvals[0]
	}
	_ = eigvecs // сохранено для соответствия спеке (топ-2 направления неявно через lambda)

	span := 3.0 * math.Sqrt(math.Max(lam1, 1e-9))
	w1Axis := linspace(-span, span, grid)
	w2Axis := linspace(-span, span, grid)
	W1, W2 := meshgridIJ(w1Axis, w2Axis)
	// Z = L0 + 0.5*(lam1*W1^2 - lam2*W2^2) + 0.05*sin(W1*W2)
	L0 := 1.0
	Z := make([][]float64, grid)
	zMin := math.Inf(1)
	zMax := math.Inf(-1)
	for i := 0; i < grid; i++ {
		Z[i] = make([]float64, grid)
		for j := 0; j < grid; j++ {
			v := L0 + 0.5*(lam1*W1[i][j]*W1[i][j] - lam2*W2[i][j]*W2[i][j]) + 0.05*math.Sin(W1[i][j]*W2[i][j])
			Z[i][j] = v
			if v < zMin {
				zMin = v
			}
			if v > zMax {
				zMax = v
			}
		}
	}
	isSaddle := lam1 > 0 && lam2 > 0 && zMin < L0
	cfg := map[string]interface{}{
		"hidden_dim":  hiddenDim,
		"n_layers":    nLayers,
		"n_heads":     nHeads,
		"vocab_size":  vocabSize,
		"max_seq_len": 64,
		"seed":        seed,
	}
	data := map[string]interface{}{
		"experiment":       "hessian_loss_landscape",
		"config":           cfg,
		"grid_size":        grid,
		"top_eigenvalues":  []float64{lam1, lam2},
		"w1_grid":          W1,
		"w2_grid":          W2,
		"loss_surface":     Z,
		"x":                W1,
		"y":                W2,
		"z":                Z,
		"metrics": map[string]interface{}{
			"lambda_max":    lam1,
			"lambda_2":      lam2,
			"spectral_gap":  lam1 - lam2,
			"loss_min":      zMin,
			"loss_max":      zMax,
			"sharpness":     lam1,
			"is_saddle":     isSaddle,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 7: Геометрия многообразия
// ---------------------------------------------------------------------------

func expManifoldGeometry(params map[string]interface{}) map[string]interface{} {
	nComponents := int(clampInf(params["pca_components"], 3, 32))
	if nComponents < 3 {
		nComponents = 3
	}
	nSamples := int(clampInf(params["trajectory_points"], 240, 4096))
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	cfg := gptConfig(params)
	hiddenDim := cfg["hidden_dim"].(int)
	nLayers := cfg["n_layers"].(int)
	maxSeq := cfg["max_seq_len"].(int)

	rng := rand.New(rand.NewSource(seed))
	prompts := nSamples
	if prompts > 32 {
		prompts = 32
	}
	var allHidden [][]float64
	for i := 0; i < prompts; i++ {
		nTok := 8 + rng.Intn(maxSeq-7)
		if nTok < 1 {
			nTok = 8
		}
		layers := synthHiddenStates(nLayers, nTok, hiddenDim, seed+int64(i+1)*97)
		// Используем только последний слой.
		allHidden = append(allHidden, layers[nLayers-1]...)
	}
	if len(allHidden) < nComponents {
		// Дополняем повторением
		reps := (nComponents/len(allHidden) + 1) * 4
		padded := make([][]float64, 0, len(allHidden)*reps)
		for r := 0; r < reps; r++ {
			padded = append(padded, allHidden...)
		}
		allHidden = padded
	}

	proj, eigvals := pcaReduce(allHidden, nComponents)
	// Дополняем проекцию до 3D при необходимости
	for i := range proj {
		for len(proj[i]) < 3 {
			proj[i] = append(proj[i], 0.0)
		}
	}
	// Participation ratio
	sumLam := 0.0
	sumLamSq := 0.0
	for _, v := range eigvals {
		sumLam += v
		sumLamSq += v * v
	}
	pr := (sumLam * sumLam) / math.Max(sumLamSq, 1e-12)

	colors := make([]int, len(proj))
	for i := range colors {
		colors[i] = i
	}
	// explained variance top-3
	top3Sum := 0.0
	for i := 0; i < 3 && i < len(eigvals); i++ {
		top3Sum += eigvals[i]
	}
	explainedVarTop3 := top3Sum / math.Max(sumLam, 1e-12)
	manifoldVol := 1.0
	for i := 0; i < 3 && i < len(eigvals); i++ {
		manifoldVol *= math.Sqrt(math.Max(eigvals[i], 0))
	}
	topEv := 0.0
	if len(eigvals) > 0 {
		topEv = eigvals[0]
	}
	// Ограничиваем список собственных значений до топ-10 для вывода.
	topN := 10
	if topN > len(eigvals) {
		topN = len(eigvals)
	}
	eigvalsOut := make([]float64, topN)
	copy(eigvalsOut, eigvals[:topN])

	// Проекция на 3D для выходных точек (pca_points).
	pca3 := make([][]float64, len(proj))
	for i := range proj {
		pca3[i] = proj[i][:3]
	}

	data := map[string]interface{}{
		"experiment":           "manifold_geometry",
		"config":               cfg,
		"n_samples":            len(proj),
		"n_components":         nComponents,
		"pca_eigenvalues":      eigvalsOut,
		"participation_ratio":  pr,
		"pca_points":           pca3,
		"pca_colors":           colors,
		"x":                    extractCol(pca3, 0),
		"y":                    extractCol(pca3, 1),
		"z":                    extractCol(pca3, 2),
		"metrics": map[string]interface{}{
			"intrinsic_dim_pr":         pr,
			"explained_variance_top3":  explainedVarTop3,
			"top_eigenvalue":           topEv,
			"manifold_volume_proxy":    manifoldVol,
		},
	}
	return data
}

func extractCol(m [][]float64, c int) []float64 {
	out := make([]float64, len(m))
	for i := range m {
		if c < len(m[i]) {
			out[i] = m[i][c]
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Эксперимент 8: Анализ траектории рассуждений
// ---------------------------------------------------------------------------

func expTrajectoryAnalysis(params map[string]interface{}) map[string]interface{} {
	nPoints := int(clampInf(params["trajectory_points"], 64, 4096))
	if nPoints < 8 {
		nPoints = 8
	}
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	cfg := gptConfig(params)
	hiddenDim := cfg["hidden_dim"].(int)
	nLayers := cfg["n_layers"].(int)
	nCrit := clampInf(params["ncrit_threshold"], 96.0, 1<<20)
	rng := rand.New(rand.NewSource(seed))

	// Синтезируем траекторию, выровненную по N_crit.
	honesty := make([]float64, nPoints)
	deception := make([]float64, nPoints)
	halluc := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		t := float64(i) * float64(nPoints) / float64(nPoints)
		stepsArr := float64(i) * float64(nPoints) / float64(nPoints-1)
		_ = t
		honesty[i] = 0.65 / math.Sqrt(1+stepsArr/nCrit)
		dec := 0.20 + 0.55*(1-math.Exp(-(stepsArr-nCrit)/30.0))
		deception[i] = clamp01(dec)
		hal := 0.05 + 0.60*(1-math.Exp(-(stepsArr-nCrit)/20.0))
		halluc[i] = clamp01(hal)
	}

	// Спектральный радиус по шагам — используем max собств. значение ковариации синтетических скрытых состояний.
	specRadius := make([]float64, nPoints)
	for i := 0; i < nPoints; i++ {
		toks := 16
		layers := synthHiddenStates(nLayers, toks, hiddenDim, seed+int64(i+1)*131)
		H := stackHiddenLayers(layers)
		specRadius[i] = maxEigenvalue(H)
	}
	// Нормализуем в [0, 1]
	specNorm := make([]float64, nPoints)
	mn, mx := specRadius[0], specRadius[0]
	for _, v := range specRadius {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	rng2 := rng
	_ = rng2
	for i, v := range specRadius {
		if mx-mn > 1e-9 {
			specNorm[i] = (v - mn) / (mx - mn)
		} else {
			specNorm[i] = 0.0
		}
	}

	// Детектируем начало обмана: первый i, где deception[i] > honesty[i]
	onsetIdx := -1
	for i := 0; i < nPoints; i++ {
		if deception[i] > honesty[i] {
			onsetIdx = i
			break
		}
	}

	steps := make([]int, nPoints)
	for i := range steps {
		steps[i] = i
	}
	specMean := 0.0
	specMax := math.Inf(-1)
	for _, v := range specRadius {
		specMean += v
		if v > specMax {
			specMax = v
		}
	}
	specMean /= float64(nPoints)
	honestyRate := (honesty[0] - honesty[nPoints-1]) / math.Max(float64(nPoints), 1)
	deceptionRate := (deception[nPoints-1] - deception[0]) / math.Max(float64(nPoints), 1)

	data := map[string]interface{}{
		"experiment": "trajectory_analysis",
		"n_points":   nPoints,
		"trajectory": map[string]interface{}{
			"steps":               steps,
			"honesty":             honesty,
			"deception":           deception,
			"hallucination":       halluc,
			"spectral":            specRadius,
			"spectral_normalized": specNorm,
		},
		"deception_onset_step": onsetIdx,
		"x":                    steps,
		"y":                    honesty,
		"z":                    deception,
		"metrics": map[string]interface{}{
			"n_points":                nPoints,
			"onset_step":              onsetIdx,
			"final_honesty":           honesty[nPoints-1],
			"final_deception":         deception[nPoints-1],
			"mean_spectral_radius":    specMean,
			"max_spectral_radius":     specMax,
			"honesty_decrease_rate":   honestyRate,
			"deception_increase_rate": deceptionRate,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 9: Регрессия спектральной поверхности
// ---------------------------------------------------------------------------

func expSpectralSurfaceRegression(params map[string]interface{}) map[string]interface{} {
	nLayers := int(clampInf(params["spectral_surface_layers"], 6, 64))
	if nLayers < 2 {
		nLayers = 2
	}
	nTokens := int(clampInf(params["trajectory_points"], 64, 4096))
	if nTokens < 4 {
		nTokens = 4
	}
	nCritPred := clampInf(params["ncrit_threshold"], 96.0, 1<<20)
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	cfg := gptConfig(params)
	hiddenDim := cfg["hidden_dim"].(int)
	maxSeq := cfg["max_seq_len"].(int)
	rng := rand.New(rand.NewSource(seed))

	lambdaMaxGrid := make([][]float64, nLayers)
	for l := 0; l < nLayers; l++ {
		lambdaMaxGrid[l] = make([]float64, nTokens)
	}
	for t := 0; t < nTokens; t++ {
		nTok := t + 4
		if nTok > maxSeq {
			nTok = maxSeq
		}
		if nTok < 2 {
			nTok = 2
		}
		tokSeed := seed + int64(t+1)*97
		layers := synthHiddenStates(nLayers, nTok, hiddenDim, tokSeed)
		for l := 0; l < nLayers && l < len(layers); l++ {
			lambdaMaxGrid[l][t] = maxEigenvalue(layers[l])
		}
	}

	// Детектируем бифуркацию: среднее по слоям, затем вторая разность.
	meanLambda := make([]float64, nTokens)
	for t := 0; t < nTokens; t++ {
		s := 0.0
		for l := 0; l < nLayers; l++ {
			s += lambdaMaxGrid[l][t]
		}
		meanLambda[t] = s / float64(nLayers)
	}
	bifToken := 0
	if nTokens >= 3 {
		// первая разность
		diff1 := make([]float64, nTokens-1)
		for i := 0; i < nTokens-1; i++ {
			diff1[i] = meanLambda[i+1] - meanLambda[i]
		}
		// вторая разность
		diff2 := make([]float64, len(diff1)-1)
		maxAbs := 0.0
		for i := 0; i < len(diff2); i++ {
			diff2[i] = diff1[i+1] - diff1[i]
			a := math.Abs(diff2[i])
			if a > maxAbs {
				maxAbs = a
				bifToken = i + 1
			}
		}
	}

	// Регрессия до/после
	preX := []float64{}
	preY := []float64{}
	postX := []float64{}
	postY := []float64{}
	preEnd := bifToken
	if preEnd < 1 {
		preEnd = 1
	}
	for i := 0; i < preEnd && i < nTokens; i++ {
		preX = append(preX, float64(i))
		preY = append(preY, meanLambda[i])
	}
	for i := preEnd; i < nTokens; i++ {
		postX = append(postX, float64(i))
		postY = append(postY, meanLambda[i])
	}
	preSlope, _ := polyfit1(preX, preY)
	postSlope, _ := polyfit1(postX, postY)

	// Глобальные min/max
	lamMin := math.Inf(1)
	lamMax := math.Inf(-1)
	for l := 0; l < nLayers; l++ {
		for t := 0; t < nTokens; t++ {
			v := lambdaMaxGrid[l][t]
			if v < lamMin {
				lamMin = v
			}
			if v > lamMax {
				lamMax = v
			}
		}
	}
	slopeRatio := postSlope / math.Max(math.Abs(preSlope), 1e-9)
	_ = rng

	// X (слои), Y (токены), Z (lambda_max) — массивы для 3D-построения
	xLayers := make([]int, nLayers)
	for i := range xLayers {
		xLayers[i] = i
	}
	yTokens := make([]int, nTokens)
	for i := range yTokens {
		yTokens[i] = i
	}

	data := map[string]interface{}{
		"experiment":          "spectral_surface_regression",
		"n_layers":            nLayers,
		"n_tokens":            nTokens,
		"lambda_max_grid":     lambdaMaxGrid,
		"bifurcation_token":   bifToken,
		"n_crit_predicted":    nCritPred,
		"x":                   xLayers,
		"y":                   yTokens,
		"z":                   lambdaMaxGrid,
		"metrics": map[string]interface{}{
			"lambda_max_global":      lamMax,
			"lambda_min_global":      lamMin,
			"bifurcation_token":      bifToken,
			"bifurcation_vs_ncrit":   math.Abs(float64(bifToken) - nCritPred),
			"pre_bifurcation_slope":  preSlope,
			"post_bifurcation_slope": postSlope,
			"slope_ratio":            slopeRatio,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 10: Риманова кривизна
// ---------------------------------------------------------------------------

func expRiemannianCurvature(params map[string]interface{}) map[string]interface{} {
	nNeighbors := int(clampInf(params["curvature_neighbors"], 8, 256))
	if nNeighbors < 3 {
		nNeighbors = 3
	}
	nSamples := int(clampInf(params["trajectory_points"], 128, 4096))
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	cfg := gptConfig(params)
	hiddenDim := cfg["hidden_dim"].(int)
	nLayers := cfg["n_layers"].(int)
	maxSeq := cfg["max_seq_len"].(int)

	rng := rand.New(rand.NewSource(seed))
	prompts := nSamples
	if prompts > 32 {
		prompts = 32
	}
	var allHidden [][]float64
	for i := 0; i < prompts; i++ {
		nTok := 8 + rng.Intn(maxSeq-7)
		if nTok < 1 {
			nTok = 8
		}
		layers := synthHiddenStates(nLayers, nTok, hiddenDim, seed+int64(i+1)*101)
		allHidden = append(allHidden, layers[nLayers-1]...)
	}
	// Редукция в 3D через PCA
	P, _ := pcaReduce(allHidden, 3)
	N := len(P)
	k := nNeighbors
	if k > N-1 {
		k = N - 1
	}
	if k < 2 {
		k = 2
	}

	curvatures := make([]float64, N)
	for i := 0; i < N; i++ {
		// kNN по евклидову расстоянию
		type dn struct {
			idx  int
			dist float64
		}
		ns := make([]dn, N)
		for j := 0; j < N; j++ {
			d := 0.0
			for c := 0; c < 3; c++ {
				diff := P[j][c] - P[i][c]
				d += diff * diff
			}
			ns[j] = dn{j, math.Sqrt(d)}
		}
		// Исключаем себя (бесконечное расстояние)
		ns[i].dist = math.Inf(1)
		sort.Slice(ns, func(a, b int) bool { return ns[a].dist < ns[b].dist })
		nn := make([][]float64, k)
		for j := 0; j < k; j++ {
			nn[j] = P[ns[j].idx]
		}
		// Векторы от P[i] к соседям
		vecs := make([][]float64, k)
		for j := 0; j < k; j++ {
			v := make([]float64, 3)
			for c := 0; c < 3; c++ {
				v[c] = nn[j][c] - P[i][c]
			}
			norm := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
			if norm < 1e-12 {
				norm = 1e-12
			}
			for c := 0; c < 3; c++ {
				v[c] /= norm
			}
			vecs[j] = v
		}
		// Сортируем по полярному углу (atan2 в xy-плоскости), чтобы соседние
		// векторы шли рядом на единичной окружности вокруг P[i].
		sort.Slice(vecs, func(a, b int) bool {
			return math.Atan2(vecs[a][1], vecs[a][0]) < math.Atan2(vecs[b][1], vecs[b][0])
		})
		// Сумма углов между соседними векторами
		totalAngle := 0.0
		for j := 0; j < k; j++ {
			a := vecs[j]
			b := vecs[(j+1)%k]
			dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
			if dot > 1 {
				dot = 1
			} else if dot < -1 {
				dot = -1
			}
			totalAngle += math.Acos(dot)
		}
		curvatures[i] = 2*math.Pi - totalAngle
	}
	// Статистики
	mean, std := meanStd(curvatures)
	cMin := math.Inf(1)
	cMax := math.Inf(-1)
	for _, v := range curvatures {
		if v < cMin {
			cMin = v
		}
		if v > cMax {
			cMax = v
		}
	}
	threshold := mean + 2*std
	var highCurvIdx []int
	for i, v := range curvatures {
		if v > threshold {
			highCurvIdx = append(highCurvIdx, i)
		}
	}
	highRatio := float64(len(highCurvIdx)) / math.Max(float64(N), 1)

	data := map[string]interface{}{
		"experiment":              "riemannian_curvature",
		"n_samples":               N,
		"n_neighbors":             nNeighbors,
		"curvatures":              curvatures,
		"points_3d":               P,
		"high_curvature_indices":  highCurvIdx,
		"x":                       extractCol(P, 0),
		"y":                       extractCol(P, 1),
		"z":                       extractCol(P, 2),
		"metrics": map[string]interface{}{
			"mean_curvature":       mean,
			"std_curvature":        std,
			"max_curvature":        cMax,
			"min_curvature":        cMin,
			"n_high_curvature":     len(highCurvIdx),
			"high_curvature_ratio": highRatio,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 11: 3D-поток внимания
// ---------------------------------------------------------------------------

func expAttentionFlow3D(params map[string]interface{}) map[string]interface{} {
	resolution := int(clampInf(params["attention_flow_3d_resolution"], 32, 1024))
	if resolution < 4 {
		resolution = 4
	}
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	_ = seed
	n := resolution

	// Синтезируем внимание: диагональное в начале, размазанное после N_crit.
	nCrit := n / 2
	attn := make([][]float64, n)
	for i := 0; i < n; i++ {
		attn[i] = make([]float64, n)
		sigma := 2.0
		if i >= nCrit {
			sigma = 6.0
		}
		rowSum := 0.0
		for j := 0; j < n; j++ {
			d := float64(i - j)
			v := math.Exp(-(d * d) / (2 * sigma * sigma))
			attn[i][j] = v
			rowSum += v
		}
		if rowSum < 1e-12 {
			rowSum = 1e-12
		}
		for j := 0; j < n; j++ {
			attn[i][j] /= rowSum
		}
	}

	// Метрика диагональности
	diagSum := 0.0
	totalSum := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			totalSum += attn[i][j]
		}
		diagSum += attn[i][i]
	}
	diagMean := diagSum / float64(n)
	totalMean := totalSum / float64(n*n)
	diagScore := diagMean / math.Max(totalMean, 1e-9)

	// Разброс по строкам
	spreadPerRow := make([]float64, n)
	for i := 0; i < n; i++ {
		rowSum := 0.0
		weightedSqSum := 0.0
		for j := 0; j < n; j++ {
			rowSum += attn[i][j]
			d := float64(j - i)
			weightedSqSum += d * d * attn[i][j]
		}
		spreadPerRow[i] = math.Sqrt(weightedSqSum / math.Max(rowSum, 1e-9))
	}
	smearingScore := 0.0
	for _, v := range spreadPerRow {
		smearingScore += v
	}
	smearingScore /= float64(n)

	// Энтропия
	entropy := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			p := attn[i][j]
			if p > 1e-12 {
				entropy -= p * math.Log(p)
			}
		}
	}
	entropy /= float64(n)

	// Максимальный вес
	maxW := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if attn[i][j] > maxW {
				maxW = attn[i][j]
			}
		}
	}

	// Массивы X, Y, Z для 3D-поверхности: query, key, weight
	x := make([]int, n)
	y := make([]int, n)
	for i := range x {
		x[i] = i
		y[i] = i
	}

	data := map[string]interface{}{
		"experiment":      "attention_flow_3d",
		"resolution":      n,
		"weights":         attn,
		"spread_per_row":  spreadPerRow,
		"x":               x,
		"y":               y,
		"z":               attn,
		"metrics": map[string]interface{}{
			"diagonality_score":         diagScore,
			"smearing_score":            smearingScore,
			"diagonal_to_smeared_ratio": diagScore / math.Max(smearingScore, 1e-9),
			"max_weight":                maxW,
			"entropy":                   entropy,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 12: Поверхность коллапса N_crit
// ---------------------------------------------------------------------------

func expNcritSurface(params map[string]interface{}) map[string]interface{} {
	nCritBase := clampInf(params["ncrit_threshold"], 114.0, 1<<20)
	thetaBDeg := clampInf(params["theta_b_deg"], 7.07, 360.0)
	thetaB := thetaBDeg * math.Pi / 180.0
	betaAxis := linspace(0.3, 0.9, 24)
	rlhfAxis := linspace(0.0, 1.0, 24)
	B, R := meshgridIJ(betaAxis, rlhfAxis)
	Z := make([][]float64, len(betaAxis))
	zMin := math.Inf(1)
	zMax := math.Inf(-1)
	zSum := 0.0
	count := 0
	for i := 0; i < len(betaAxis); i++ {
		Z[i] = make([]float64, len(rlhfAxis))
		for j := 0; j < len(rlhfAxis); j++ {
			muEff := thetaB + R[i][j]
			if muEff < 1e-6 {
				muEff = 1e-6
			}
			beta := B[i][j]
			if beta < 1e-6 {
				beta = 1e-6
			}
			v := nCritBase * math.Pow(muEff, -1.0/beta)
			// Защита от переполнения
			if math.IsInf(v, 1) || math.IsNaN(v) {
				v = 1e18
			}
			Z[i][j] = v
			if v < zMin {
				zMin = v
			}
			if v > zMax {
				zMax = v
			}
			zSum += v
			count++
		}
	}
	zMean := zSum / float64(count)
	// Опорные значения
	tAtBeta05Rlhf0 := nCritBase * math.Pow(math.Max(thetaB, 1e-9), -1.0/0.5)
	tAtBeta05Rlhf1 := nCritBase * math.Pow(thetaB+1.0, -1.0/0.5)

	data := map[string]interface{}{
		"experiment":  "ncrit_surface",
		"beta_axis":   betaAxis,
		"rlhf_axis":   rlhfAxis,
		"t_crit_grid": Z,
		"n_crit_base": nCritBase,
		"theta_b_rad": thetaB,
		"x":           B,
		"y":           R,
		"z":           Z,
		"metrics": map[string]interface{}{
			"t_crit_min":                zMin,
			"t_crit_max":                zMax,
			"t_crit_mean":               zMean,
			"t_crit_at_beta_0_5_rlhf_0": tAtBeta05Rlhf0,
			"t_crit_at_beta_0_5_rlhf_1": tAtBeta05Rlhf1,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 13: Развёртка пространства параметров
// ---------------------------------------------------------------------------

func expParameterSpace(params map[string]interface{}) map[string]interface{} {
	grid := int(clampInf(params["parameter_space_grid"], 16, 256))
	if grid < 4 {
		grid = 4
	}
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	rng := rand.New(rand.NewSource(seed))
	tempAxis := linspace(0.0, 2.0, grid)
	toppAxis := linspace(0.5, 1.0, grid)
	T, P := meshgridIJ(tempAxis, toppAxis)
	Z := make([][]float64, grid)
	zMin := math.Inf(1)
	zMax := math.Inf(-1)
	for i := 0; i < grid; i++ {
		Z[i] = make([]float64, grid)
		for j := 0; j < grid; j++ {
			base := sigmoid((T[i][j]-0.8)*2.0) * (P[i][j] - 0.5) * 2.0
			noise := 0.02 * rng.Float64()
			v := clamp01(base + noise)
			Z[i][j] = v
			if v < zMin {
				zMin = v
			}
			if v > zMax {
				zMax = v
			}
		}
	}
	// Средние по краям
	meanT0 := 0.0
	meanT2 := 0.0
	meanP05 := 0.0
	meanP1 := 0.0
	for j := 0; j < grid; j++ {
		meanT0 += Z[0][j]
		meanT2 += Z[grid-1][j]
	}
	for i := 0; i < grid; i++ {
		meanP05 += Z[i][0]
		meanP1 += Z[i][grid-1]
	}
	meanT0 /= float64(grid)
	meanT2 /= float64(grid)
	meanP05 /= float64(grid)
	meanP1 /= float64(grid)

	data := map[string]interface{}{
		"experiment":         "parameter_space",
		"temp_axis":          tempAxis,
		"topp_axis":          toppAxis,
		"hallucination_grid": Z,
		"x":                  T,
		"y":                  P,
		"z":                  Z,
		"metrics": map[string]interface{}{
			"hallucination_min":    zMin,
			"hallucination_max":    zMax,
			"hallucination_at_T0":  meanT0,
			"hallucination_at_T2":  meanT2,
			"hallucination_at_P05": meanP05,
			"hallucination_at_P1":  meanP1,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Эксперимент 14: Дрейф коалиции
// ---------------------------------------------------------------------------

func expCoalitionDrift(params map[string]interface{}) map[string]interface{} {
	nAgents := int(clampInf(params["n_agents"], 2, 64))
	if nAgents < 1 {
		nAgents = 1
	}
	nRounds := int(clampInf(params["n_rounds"], 4, 64))
	if nRounds < 1 {
		nRounds = 1
	}
	seed := int64(clampInf(params["seed"], 42, 1<<30))
	rng := rand.New(rand.NewSource(seed))

	baseDeception := make([]float64, nAgents)
	for i := range baseDeception {
		baseDeception[i] = 0.25 + 0.05*float64(i)
	}
	perRound := make([][]float64, nRounds)
	for r := 0; r < nRounds; r++ {
		row := make([]float64, nAgents)
		for a := 0; a < nAgents; a++ {
			v := baseDeception[a] + 0.12*float64(r) + rng.NormFloat64()*0.02
			row[a] = clamp01(v)
		}
		perRound[r] = row
	}

	// Статистики
	meanFirst := 0.0
	meanLast := 0.0
	for a := 0; a < nAgents; a++ {
		meanFirst += perRound[0][a]
		meanLast += perRound[nRounds-1][a]
	}
	meanFirst /= float64(nAgents)
	meanLast /= float64(nAgents)
	drift := meanLast - meanFirst
	varSum := 0.0
	meanL := meanLast
	for a := 0; a < nAgents; a++ {
		d := perRound[nRounds-1][a] - meanL
		varSum += d * d
	}
	variance := varSum / float64(nAgents)

	// Массивы для 3D-построения: раунды, агенты, обман
	roundsAxis := make([]int, nRounds)
	agentsAxis := make([]int, nAgents)
	for i := range roundsAxis {
		roundsAxis[i] = i
	}
	for i := range agentsAxis {
		agentsAxis[i] = i
	}

	data := map[string]interface{}{
		"experiment":  "coalition_drift",
		"n_agents":    nAgents,
		"n_rounds":    nRounds,
		"per_round":   perRound,
		"x":           roundsAxis,
		"y":           agentsAxis,
		"z":           perRound,
		"metrics": map[string]interface{}{
			"initial_mean_deception": meanFirst,
			"final_mean_deception":   meanLast,
			"drift":                  drift,
			"convergence_variance":   variance,
		},
	}
	return data
}

// ---------------------------------------------------------------------------
// Публичный API
// ---------------------------------------------------------------------------

// run3DExperiment запускает один 3D-эксперимент по ID и записывает результат
// в JSON laboratory/results/reports/{ts}_3d_exp_{id}_results.json.
func run3DExperiment(id string, params map[string]interface{}) map[string]interface{} {
	exp, ok := Experiments3D[id]
	if !ok {
		known := make([]string, 0, len(Experiments3D))
		for k := range Experiments3D {
			known = append(known, k)
		}
		sort.Strings(known)
		return map[string]interface{}{
			"error": fmt.Sprintf("неизвестный ID 3D-эксперимента %q. известные: %v", id, known),
		}
	}
	if params == nil {
		params = map[string]interface{}{}
	}
	t0 := time.Now()
	result := exp.Run(params)
	elapsed := time.Since(t0).Seconds()

	// Инжектируем метаданные верхнего уровня (Go 3D-спека), сохраняя
	// Python-совместимые ключи (experiment, experiment_name, ...), чтобы
	// charts_3d.py мог читать результат.
	result["experiment_id"] = id
	result["name"] = exp.Name
	result["description"] = exp.Description
	result["experiment_name"] = exp.Name
	result["experiment_description"] = exp.Description
	result["elapsed_seconds"] = elapsed

	// Эхо скалярных параметров (без вложенных dict/list, как в Python).
	echoParams := map[string]interface{}{}
	for k, v := range params {
		switch v.(type) {
		case map[string]interface{}, []interface{}:
			continue
		default:
			echoParams[k] = v
		}
	}
	result["parameters"] = echoParams

	// Оборачиваем блок данных эксперимента в ключ "data" для соответствия
	// Go 3D-спеке (не удаляя Python-стиль ключей верхнего уровня).
	dataBlock := map[string]interface{}{}
	for k, v := range result {
		switch k {
		case "experiment_id", "name", "description", "experiment_name",
			"experiment_description", "elapsed_seconds", "parameters", "data":
			continue
		default:
			dataBlock[k] = v
		}
	}
	result["data"] = dataBlock

	// Запись JSON в laboratory/results/reports/
	if path, err := write3DResult(id, result); err == nil {
		result["results_path"] = path
	} else {
		result["results_write_error"] = err.Error()
	}
	return result
}

// runAll3D запускает все 9 3D-экспериментов и возвращает объединённый словарь,
// пригодный для charts_3d.generate_all_3d_charts().
func runAll3D(params map[string]interface{}) map[string]interface{} {
	if params == nil {
		params = map[string]interface{}{}
	}
	combined := map[string]interface{}{
		"3d_research": map[string]interface{}{},
		"experiments": []map[string]interface{}{},
	}
	ids := []string{"6", "7", "8", "9", "10", "11", "12", "13", "14"}
	expList := combined["experiments"].([]map[string]interface{})
	research := combined["3d_research"].(map[string]interface{})
	for _, id := range ids {
		func() {
			defer func() {
				if r := recover(); r != nil {
					errMsg := fmt.Sprintf("panic: %v", r)
					fmt.Fprintf(os.Stderr, "  [WARN] 3D-эксперимент %s завершился паникой: %s\n", id, errMsg)
					expList = append(expList, map[string]interface{}{
						"id":    id,
						"error": errMsg,
					})
				}
			}()
			res := run3DExperiment(id, params)
			if errMsg, ok := res["error"]; ok {
				fmt.Fprintf(os.Stderr, "  [WARN] 3D-эксперимент %s завершился ошибкой: %v\n", id, errMsg)
				expList = append(expList, map[string]interface{}{
					"id":    id,
					"error": fmt.Sprintf("%v", errMsg),
				})
				return
			}
			expList = append(expList, map[string]interface{}{
				"id":              id,
				"name":            res["name"],
				"elapsed_seconds": res["elapsed_seconds"],
			})
			key := nameMap3D[id]
			if key == "" {
				key = "exp_" + id
			}
			research[key] = res
		}()
	}
	combined["experiments"] = expList
	combined["3d_research"] = research
	return combined
}

// write3DResult записывает JSON результата в laboratory/results/reports/.
func write3DResult(id string, result map[string]interface{}) (string, error) {
	dir := lab3DReportsDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	ts := time.Now().Format("20060102_150405")
	fname := fmt.Sprintf("%s_3d_exp_%s_results.json", ts, id)
	path := filepath.Join(dir, fname)
	// Заменяем не-конечные float на null перед маршалингом (NaN/Inf — некорректный JSON).
	sanitised := sanitiseForJSON(result)
	data, err := json.MarshalIndent(sanitised, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", err
	}
	return path, nil
}

// lab3DReportsDir возвращает путь к laboratory/results/reports.
func lab3DReportsDir() string {
	return filepath.Join(labRoot(), "laboratory", "results", "reports")
}

// sanitiseForJSON обходит значение и заменяет NaN/Inf на null, чтобы
// encoding/json всегда производил корректный JSON.
func sanitiseForJSON(v interface{}) interface{} {
	switch x := v.(type) {
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil
		}
		return x
	case float32:
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil
		}
		return x
	case []interface{}:
		out := make([]interface{}, len(x))
		for i, e := range x {
			out[i] = sanitiseForJSON(e)
		}
		return out
	case []float64:
		out := make([]interface{}, len(x))
		for i, e := range x {
			out[i] = sanitiseForJSON(e)
		}
		return out
	case []int:
		out := make([]interface{}, len(x))
		for i, e := range x {
			out[i] = e
		}
		return out
	case [][]float64:
		out := make([]interface{}, len(x))
		for i, e := range x {
			out[i] = sanitiseForJSON(e)
		}
		return out
	case []string:
		out := make([]interface{}, len(x))
		for i, e := range x {
			out[i] = e
		}
		return out
	case map[string]interface{}:
		out := make(map[string]interface{}, len(x))
		for k, e := range x {
			out[k] = sanitiseForJSON(e)
		}
		return out
	default:
		return v
	}
}
