//https://aria42.com/blog/2014/12/understanding-lbfgs
#define _CRT_SECURE_NO_WARNINGS
#define USE_LBFGS

#include <omp.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <deque>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
// Using updated (v2) interfaces for CUBLAS and CUSPARSE
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "helper_cuda.h"  // helper function CUDA error checking and initialization
#include "helper_functions.h" // helper for shared functions common to CUDA Samples

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "array2.cuh"

#define THREADS_PER_BLOCK 512

namespace cg = cooperative_groups;

std::vector<int> objEqHeads, gradEqHeads;
std::vector<double> objEqVals, gradEqVals;

enum NodeType {
	NODE_CONST,
	NODE_OPER,
	NODE_VAR
};

enum OpType {
	OP_PLUS = 0,
	OP_MINUS = 1,
	OP_UMINUS = 2,
	OP_TIME = 3,
	OP_DIVIDE = 4,
	OP_SIN,
	OP_COS,
	OP_TG,
	OP_CTG,
	OP_SEC,
	OP_CSC,
	OP_ARCSIN,
	OP_ARCCOS,
	OP_ARCTG,
	OP_ARCCTG,
	OP_POW,
	OP_EXP,
	OP_EEXP,
	OP_SQR,
	OP_SQRT,
	OP_LOG,
	OP_LN,
	OP_NULL = -1
};

typedef struct _EqInfo {
	NodeType _type;
	double _val;
	int _var;
	OpType _op;
	int _left;
	int _right;
} EqInfo;

// enum VarType {
// 	VAR_CONST,
// 	VAR_UNSOLVED,
// 	VAR_SOLVED,
// 	VAR_DELETED,
// 	VAR_FREE
// };

enum OptimType {
	BFGS,
	LBFGS
};

// struct VarInfo {
// 	VarType	_type;
// 	double		_val;

// 	VarInfo(VarType ty, double val) : _type(ty), _val(val) {}
// };

#define epsZero1 1e-20
#define epsZero2 1e-7
#ifndef M_PI_2
#define M_PI_2 (1.57079632679489661923)
#endif


#define		BFGS_MAXIT	500
#define		BFGS_STEP	0.1

static int _GetMaxIt()
{
	return BFGS_MAXIT;
}

static double _GetStep()
{
	return BFGS_STEP;
}

static double _GetEps()
{
	return 0.01;
}

static void _ConstructVarTab(std::vector<double>& vars, std::vector<int>& varMap, std::vector<int>& revMap);
static void _ConstructObjEqTab(std::vector<EqInfo>& eqs, int& numEqs, const std::vector<int>& revMap);
static void _ConstructGradEqTab(std::vector<EqInfo>& eqs, int& numEqs, const std::vector<int>& revMap);
static void _ScatterVarTab(std::vector<double>& x, std::vector<int>& varMap);

static void _VecCopy(std::vector<double>& dst, const std::vector<double>& src)
{
	int n = src.size();
	for (int i = 0; i < n; i++)
		dst[i] = src[i];
}

static void _VecSub(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& ret)
{
	int n = a.size();
	for (int i = 0; i < n; i++)
		ret[i] = a[i] - b[i];
}

static double _VecDot(const std::vector<double>& a, const std::vector<double>& b)
{
	double s = 0;
	int n = a.size();
	for (int i = 0; i < n; i++)
		s += a[i] * b[i];

	return s;
}

static void _VecMult(std::vector<double>& v, double t)
{
	int n = v.size();
	for (int i = 0; i < n; i++)
		v[i] *= t;
}

static double _VecAdd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& ret)
{
	int n = a.size();
	for (int i = 0; i < n; i++)
		ret[i] = a[i] + b[i];
}

static void _VecAxPy(const std::vector<double>& a, double t, const std::vector<double>& b, std::vector<double>& ret)
{
	int n = a.size();
	for (int i = 0; i < n; i++)
		ret[i] = a[i] + b[i] * t;
}

static double _VecLen(const std::vector<double>& v)
{
	return sqrt(_VecDot(v, v));
}

static void _VecNorm(std::vector<double>& v)
{
	double tmp = _VecLen(v);
	if (tmp > 0.0) {
		_VecMult(v, 1.0 / tmp);
	}
}

static void 
_CalcEqNew2(const std::vector<double>& x, const std::vector<EqInfo>& etab, int st, int ed, std::vector<double>& vtab)
{
	for (int i = ed - 1; i >= st; i--) {
		const EqInfo& eq = etab[i];
		switch (eq._type) {
		case NODE_CONST:
			vtab[i] = eq._val;
			break;

		case NODE_VAR: {
			int idx = eq._var;
			vtab[i] = x[idx];
			break;
		}

		case NODE_OPER: {
			double left = vtab[eq._left];
			double right = vtab[eq._right];
			switch (eq._op) {
			case	OP_PLUS:
				vtab[i] = (left + right);
				break;
			case	OP_MINUS:
				vtab[i] = (left - right);
				break;
			case	OP_UMINUS:
				vtab[i] = -right;
				break;
			case	OP_TIME:
				vtab[i] = (left * right);
				break;
			case	OP_DIVIDE:
				vtab[i] = (left / right);
				break;
			case	OP_SIN:
				vtab[i] = (sin(left));
				break;
			case	OP_COS:
				vtab[i] = (cos(left));
				break;
			case	OP_TG:
				vtab[i] = (tan(left));
				break;
			case	OP_CTG:
				vtab[i] = (1.0 / tan(left));
				break;
			case	OP_SEC:
				vtab[i] = (1.0 / cos(left));
				break;
			case	OP_CSC:
				vtab[i] = (1.0 / sin(left));
				break;
			case	OP_ARCSIN:
				vtab[i] = (asin(left));
				break;
			case	OP_ARCCOS:
				vtab[i] = (acos(left));
				break;
			case	OP_ARCTG:
				vtab[i] = (atan(left));
				break;
			case	OP_ARCCTG:
				vtab[i] = (atan(-left) + M_PI_2);
				break;
			case	OP_POW:
				vtab[i] = (pow(left, right));
				break;
			case	OP_EEXP:
				vtab[i] = (exp(left));
				break;
			case	OP_EXP:
				vtab[i] = (exp(left * log(right)));
				break;
			case	OP_LN:
				vtab[i] = (log(left));
				break;
			case	OP_LOG:
				vtab[i] = (log(right) / log(left));
				break;
			case	OP_SQR:
				vtab[i] = (left * left);
				break;
			case	OP_SQRT:
				vtab[i] = (sqrt(left));
				break;
			default:
				fprintf(stderr, "Unknown operator in EsCalcTree()\n");
				assert(0);
			}
		}
		}

	}
}

static double _CalcEqNew1(const std::vector<double>& x, const EqInfo& eq, const std::vector<EqInfo>& etab,
	int item, const std::vector<int> &htab, int allNum, std::vector<double> &vtab)
{
	int ed = item < 0 ? allNum : htab[item + 1];
	int st = item < 0 ? htab[-item] : htab[item];
	_CalcEqNew2(x, etab,  st, ed, vtab);

	switch (eq._type) {
	case NODE_OPER: {
		double left = vtab[eq._left];
		double right = vtab[eq._right];
		switch (eq._op) {
		case	OP_PLUS:
			return(left + right);
		case	OP_MINUS:
			return(left - right);
		case	OP_UMINUS:
			return(-right);
		case	OP_TIME:
			return(left * right);
		case	OP_DIVIDE:
			return(left / right);
		case	OP_SIN:
			return(sin(left));
		case	OP_COS:
			return(cos(left));
		case	OP_TG:
			return(tan(left));
		case	OP_CTG:
			return(1.0 / tan(left));
		case	OP_SEC:
			return(1.0 / cos(left));
		case	OP_CSC:
			return(1.0 / sin(left));
		case	OP_ARCSIN:
			return(asin(left));
		case	OP_ARCCOS:
			return(acos(left));
		case	OP_ARCTG:
			return(atan(left));
		case	OP_ARCCTG:
			return(atan(-left) + M_PI_2);
		case	OP_POW:
			return(pow(left, right));
		case	OP_EEXP:
			return(exp(left));
		case	OP_EXP:
			return(exp(left * log(right)));
		case	OP_LN:
			return(log(left));
		case	OP_LOG:
			return(log(right) / log(left));
		case	OP_SQR:
			return(left * left);
		case	OP_SQRT:
			return(sqrt(left));
		default:
			fprintf(stderr, "Unknown operator in EsCalcTree()\n");
			assert(0);
			return  (0.0);
		}
	}
	}

	assert(0);
	return 0;
}

static double _CalcEq(const std::vector<double>& x, const EqInfo& eq, const std::vector<EqInfo>& etab)
{
	double left, right;

	switch (eq._type) {
	case NODE_CONST:
		return(eq._val);
		break;

	case NODE_VAR: {
		int idx = eq._var;
		return x[idx];
		break;
	}

	case NODE_OPER: {
		left = _CalcEq(x, etab[eq._left], etab);
		right = _CalcEq(x, etab[eq._right], etab);
		switch (eq._op) {
		case	OP_PLUS:
			return(left + right);
		case	OP_MINUS:
			return(left - right);
		case	OP_UMINUS:
			return(-right);
		case	OP_TIME:
			return(left * right);
		case	OP_DIVIDE:
			return(left / right);
		case	OP_SIN:
			return(sin(left));
		case	OP_COS:
			return(cos(left));
		case	OP_TG:
			return(tan(left));
		case	OP_CTG:
			return(1.0 / tan(left));
		case	OP_SEC:
			return(1.0 / cos(left));
		case	OP_CSC:
			return(1.0 / sin(left));
		case	OP_ARCSIN:
			return(asin(left));
		case	OP_ARCCOS:
			return(acos(left));
		case	OP_ARCTG:
			return(atan(left));
		case	OP_ARCCTG:
			return(atan(-left) + M_PI_2);
		case	OP_POW:
			return(pow(left, right));
		case	OP_EEXP:
			return(exp(left));
		case	OP_EXP:
			return(exp(left * log(right)));
		case	OP_LN:
			return(log(left));
		case	OP_LOG:
			return(log(right) / log(left));
		case	OP_SQR:
			return(left * left);
		case	OP_SQRT:
			return(sqrt(left));
		default:
			fprintf(stderr, "Unknown operator in EsCalcTree()\n");
			assert(0);
			return  (0.0);
		}
	}
	}

	assert(0);
	return 0;
}

static double _CalcObj(const std::vector<double>& x,
	const std::vector<EqInfo>& eqs, int eqNum)
{
	std::vector<double> tmp;
	tmp.resize(eqNum);

	for (int i = 0; i < eqNum; i++) {
		//double v1 = _CalcEq(x, eqs[i], eqs);
		double v2 = _CalcEqNew1(x, eqs[i], eqs, i == eqNum - 1 ? -i : i, objEqHeads, objEqVals.size(), objEqVals);
		//assert(v1 == v2);
		tmp[i] = v2;
	}

	return _VecDot(tmp, tmp);
}

static void _CalcGrad(const std::vector<double>& x, std::vector<double>& g,
	const std::vector<EqInfo>& eqs)
{
	int n = x.size();
	for (int i = 0; i < n; i++) {
		//double v1 = _CalcEq(x, eqs[i], eqs);
		double v2 = _CalcEqNew1(x, eqs[i], eqs, i == n - 1 ? -i : i, gradEqHeads, gradEqVals.size(), gradEqVals);
		//assert(v1 == v2);
		g[i] = v2;
	}
}

static double _CalcObj(const std::vector<double>& x0, double h, const std::vector<double>& p,
	const std::vector<EqInfo>& eqs, int eqNum)
{
	std::vector<double> xt;
	xt.resize(x0.size());
	_VecAxPy(x0, h, p, xt);
	return _CalcObj(xt, eqs, eqNum);
}

static void _CalcyTH(const std::vector<double>& y, const array2<double>& H, std::vector<double>& yTH)
{
	int	i, j;
	int n = y.size();

	std::fill(yTH.begin(), yTH.end(), 0.0);
	for (j = 0; j < n; j++)
		for (i = 0; i < n; i++) {
			yTH[i] += (y[j] * H(j, i));
		}
}

static void _CalcHy(const array2<double>& H, const std::vector<double>& y, std::vector<double>& Hy)
{
	int	i, j;
	int n = y.size();

	for (i = 0; i < n; i++) {
		Hy[i] = 0.0;
		for (j = 0; j < n; j++)
			Hy[i] += (y[j] * H(i, j));
	}
}

static void _Calcp(const array2<double>& H, const std::vector<double>& g, std::vector<double>& p)
{
	_CalcHy(H, g, p);

	int n = p.size();
	while (n--)
		p[n] = -p[n];
}

#define BFGS_MAXBOUND	1e+10
static void _DetermineInterval(
	const std::vector<double>& x0, double h, const std::vector<double>& p,
	double* left, double* right,
	const std::vector<EqInfo>& eqs, int eqNum)
{
	double	A, B, C, D, u, v, w, s, r;

	A = _CalcObj(x0, 0.0, p, eqs, eqNum);
	B = _CalcObj(x0, h, p, eqs, eqNum);
	if (B > A) {
		s = -h;
		C = _CalcObj(x0, s, p, eqs, eqNum);
		if (C > A) {
			*left = -h;
			*right = h;
			return;
		}
		B = C;
	}
	else {
		s = h;
	}
	u = 0.0;
	v = s;
	while (1) {
		s += s;
		if (fabs(s) > BFGS_MAXBOUND) {
			*left = *right = 0.0;
			return;
		}
		w = v + s;
		C = _CalcObj(x0, w, p, eqs, eqNum);
		if (C >= B)
			break;
		u = v;
		A = B;
		v = w;
		B = C;
	}
	r = (v + w) * 0.5;
	D = _CalcObj(x0, r, p, eqs, eqNum);
	if (s < 0.0) {
		if (D < B) {
			*left = w;
			*right = v;
		}
		else {
			*left = r;
			*right = u;
		}
	}
	else {
		if (D < B) {
			*left = v;
			*right = w;
		}
		else {
			*left = u;
			*right = r;
		}
	}
}

static void _GodenSep(
	const std::vector<double>& x0, const std::vector<double>& p,
	double left, double right, std::vector<double>& x,
	const std::vector<EqInfo>& eqs, int eqNum)
{
	static double	beta = 0.61803398874989484820;
	double			t1, t2, f1, f2;

	t2 = left + beta * (right - left);
	f2 = _CalcObj(x0, t2, p, eqs, eqNum);
ENTRY1:
	t1 = left + right - t2;
	f1 = _CalcObj(x0, t1, p, eqs, eqNum);
ENTRY2:
	if (fabs(t1 - t2) < epsZero2) {
		t1 = (t1 + t2) / 2.0;
		//printf("LineSearch t = %lf\n", t1*10000);

		_VecAxPy(x0, t1, p, x);
		return;
	}
	if ((fabs(left) > BFGS_MAXBOUND) || (fabs(left) > BFGS_MAXBOUND))
		return;
	if (f1 <= f2) {
		right = t2;
		t2 = t1;
		f2 = f1;
		goto ENTRY1;
	}
	else {
		left = t1;
		t1 = t2;
		f1 = f2;
		t2 = left + beta * (right - left);
		f2 = _CalcObj(x0, t2, p, eqs, eqNum);
		goto ENTRY2;
	}
}

static void _LinearSearch(
	const std::vector<double>& x0,
	const std::vector<double>& p,
	double h,
	std::vector<double>& x,
	const std::vector<EqInfo>& eqs,
	int eqNum)
{
	double	left, right;

	_DetermineInterval(x0, h, p, &left, &right, eqs, eqNum);
	if (left == right)
		return;

	//printf("%lf, %lf\n", left, right);
	_GodenSep(x0, p, left, right, x, eqs, eqNum);
}

#define	H_EPS1	1e-5
#define	H_EPS2	1e-5
#define	H_EPS3	1e-4

static bool _HTerminate(
	const std::vector<double>& xPrev,
	const std::vector<double>& xNow,
	double fPrev, double fNow,
	const std::vector<double>& gNow)
{
	double	ro;
	std::vector<double> xDif(xNow.size());

	if (_VecLen(gNow) >= H_EPS3)
		return false;

	_VecSub(xNow, xPrev, xDif);
	ro = _VecLen(xPrev);
	if (ro < H_EPS2)
		ro = 1.0;
	ro *= H_EPS1;
	if (_VecLen(xDif) >= ro)
		return false;

	ro = fabs(fPrev);
	if (ro < H_EPS2)
		ro = 1.0;
	ro *= H_EPS1;
	fNow -= fPrev;
	if (fabs(fNow) >= ro)
		return false;

	return true;
}

void
AnalysisEqs(const std::vector<EqInfo>& eqTab, int eqNum, std::vector<int>& eqHeads)
{
	eqHeads.resize(eqNum);
	for (int i = 0; i < eqNum; i++) {
		const EqInfo& eq = eqTab[i];
		int left = eq._left;
		int right = eq._right;

		eqHeads[i] = left;
	}
}

__device__ void gpuSpVM(double *matrix, double *vector, double *result, 
                                        int rows, int cols, double alpha,
										const cg::thread_block &cta, const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < cols; i += grid.size()) {
        double output = 0.0;
        for (int j = 0; j < rows; j++) 
		{
        	output += alpha * vector[j] * matrix[cols * j + i];
        }

        result[i] = output;
		// printf("yTH[%d] = %f\n", i, result[i]);
    }
}

__device__ void gpuSpMV(double *matrix, double *vector, double *result, 
                                        int rows, int cols, double alpha,
										const cg::thread_block &cta, const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < rows; i += grid.size()) {
        double output = 0.0;
        for (int j = 0; j < cols; j++) 
		{
        	output += alpha * vector[j] * matrix[cols * i + j];
        }

        result[i] = output;
    }
}

__device__ void gpuScaleVector(double *y, double scale, int size,
                            const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        y[i] = scale * y[i];
    }
}

__device__ void gpuSaxpy(double *x, double *y, double *r, double a, int size,
                         const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        r[i] = a * x[i] + y[i];
		// printf("r[%d] = %f\n", i, r[i]);
    }
}

__device__ void gpuDotProduct(double *vecA, double *vecB, double *result,
                              int size, const cg::thread_block &cta,
                              const cg::grid_group &grid) {
    extern __shared__ double tmp[];

    double temp_sum = 0.0;
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        temp_sum += vecA[i] * vecB[i];
    }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

    if (tile32.thread_rank() == 0) {
        tmp[tile32.meta_group_rank()] = temp_sum;
    }

    cg::sync(cta);

    if (tile32.meta_group_rank() == 0) {
        temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
        temp_sum = cg::reduce(tile32, temp_sum, cg::plus<double>());

        if (tile32.thread_rank() == 0) {
        	atomicAdd(result, temp_sum);
        }
    }
}

__device__ void gpuCopyVector(double *srcA, double *destB, int size,
                                const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        destB[i] = srcA[i];
    }
}

__device__ void gpuScaleVectorAndSaxpy(const double *x, double *y, double a, double scale, int size,
                            const cg::grid_group &grid) {
    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        y[i] = a * x[i] + scale * y[i];
    }
}

__device__ void gpuHUpdate(double *H, double *Hy, double *yTH, double tmp, double sy, double *s, int n, const cg::grid_group &grid) 
{
	for (int i = grid.thread_rank(); i < n * n; i += grid.size()) {
        int row = i / n;
		int col = i % n;
		H[i] += (((tmp * s[row] * s[col]) - Hy[row] * s[col] -
					s[row] * yTH[col]) / sy);
	}
}

extern "C" __global__ void BFGSMultiply(
    double* gPrev, double* gNow, double* xPrev, double* xNow, double* H,
    double* p, double* yTH, double* Hy, double* s, double* y, 
    double* sy, double* dot_result, int n) 
{
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    double alpha = 1.0;
    double alpham1 = -1.0;

    // Compute y = gNow - gPrev
    gpuSaxpy(gPrev, gNow, y, alpham1, n, grid);
    cg::sync(grid);

    // Compute s = xNow - xPrev
    gpuSaxpy(xPrev, xNow, s, alpham1, n, grid);
    cg::sync(grid);

    // Compute sy = dot(s, y)
    gpuDotProduct(s, y, sy, n, cta, grid);
    cg::sync(grid);

    // Proceed only if sy is above a certain threshold (epsZero1)
    if (fabs(*sy) >= epsZero1) {
        // Compute yTH = H * y
        gpuSpVM(H, y, yTH, n, n, alpha, cta, grid);
        cg::sync(grid);

        // Compute Hy = H * y
        gpuSpMV(H, y, Hy, n, n, alpha, cta, grid);
        cg::sync(grid);

        // Initialize dot_result to zero
        if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0;
        cg::sync(grid);

        // Compute dot_result = dot(yTH, y)
        gpuDotProduct(yTH, y, dot_result, n, cta, grid);
        cg::sync(grid);
        // Update H matrix with computed values
        double tmp = 1.0 + *dot_result / *sy;

        gpuHUpdate(H, Hy, yTH, tmp, *sy, s, n, grid);
        cg::sync(grid);

        // Compute p = -H * gNow
        gpuSpMV(H, gNow, p, n, n, alpham1, cta, grid);
        cg::sync(grid);

        // Re-initialize dot_result to zero before re-use
        if (threadIdx.x == 0 && blockIdx.x == 0) *dot_result = 0;
        cg::sync(grid);

        // Compute the norm of p
        gpuDotProduct(p, p, dot_result, n, cta, grid);
        cg::sync(grid);

        // Normalize p to unit length
        gpuScaleVector(p, 1.0 / sqrt(*dot_result), n, grid);
        cg::sync(grid);
    }
}


int BFGSSolveEqs()
{
	double eps = _GetEps()*_GetEps();
	int itMax = _GetMaxIt();

	double step = _GetStep();

	std::vector<double> xNow, xKeep;
	std::vector<int> varMap, revMap;
	std::vector<EqInfo> objEqs;
	int numObjEqs;
	std::vector<EqInfo> gradEqs;
	int numGradEqs;

	cudaDeviceProp deviceProp;
	int devID = 0;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	if (!deviceProp.managedMemory) {
		// This sample requires being run on a device that supports Unified Memory
		fprintf(stderr, "Unified Memory not supported on this device\n");
		exit(EXIT_WAIVED);
	}

	// This sample requires being run on a device that supports Cooperative Kernel
	// Launch
	if (!deviceProp.cooperativeLaunch) {
		printf(
				"\nSelected GPU (%d) does not support Cooperative Kernel Launch, "
				"Waiving the run\n",
				devID);
		exit(EXIT_WAIVED);
	}

	{
		FILE* fp = fopen("D:\\yzj\\bfgs\\data\\bfgs-large.dat", "rb");
		if (fp == NULL) {
			printf("bfgs.dat failed to open for read.\n");
			return false;
		}

		double t0 = omp_get_wtime();
		int nx;
		fread(&nx, sizeof(int), 1, fp);
		xNow.resize(nx);
		fread(xNow.data(), sizeof(double), nx, fp);

		int n1, no;
		fread(&n1, sizeof(int), 1, fp);
		fread(&no, sizeof(int), 1, fp);
		numObjEqs = no;
		objEqs.resize(n1);
		fread(objEqs.data(), sizeof(EqInfo), n1, fp);

		int ng;
		fread(&ng, sizeof(int), 1, fp);
		gradEqs.resize(ng);
		fread(gradEqs.data(), sizeof(EqInfo), ng, fp);
		numGradEqs = ng;

		int nk;
		fread(&nk, sizeof(int), 1, fp);
		assert(nk == nx);
		xKeep.resize(nk);
		fread(xKeep.data(), sizeof(double), nk, fp);

		double dt = omp_get_wtime() - t0;
		printf("###Data loading used %2.5f s ...\n", dt);

		//to remove recursive eval
		AnalysisEqs(objEqs, numObjEqs, objEqHeads);
		objEqVals.resize(objEqs.size());
		AnalysisEqs(gradEqs, nx, gradEqHeads);
		gradEqVals.resize(gradEqs.size());
	}

	double t0 = omp_get_wtime();
	//Do optimization
	double fNow = 0, fPrev = 0;;
	int n = xNow.size();
	int k = 0;//useless?
	int itCounter = 0;

	std::vector<double> gPrev, gNow, xPrev, p, y, s, yTH, Hy;

	array2<double> H;

	xPrev = xNow;
	gPrev.resize(n);
	gNow.resize(n);
	p.resize(n);
	y.resize(n);
	s.resize(n);
	yTH.resize(n);
	Hy.resize(n);
	H.resize(n, n);

	double *d_gnow, *d_gprev, *d_xnow, *d_xprev, *d_s, *d_y, *d_yTH, *d_Hy, *d_p, *d_H, *d_sy, *d_dot_result;
	checkCudaErrors(cudaMalloc(&d_gnow, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_gprev, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_xnow, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_xprev, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_s, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_y, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_yTH, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_Hy, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_p, sizeof(double) * n));
	checkCudaErrors(cudaMalloc(&d_H, sizeof(double) * n * n));
	checkCudaErrors(cudaMalloc(&d_sy, sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_dot_result, sizeof(double)));

	fPrev = _CalcObj(xNow, objEqs, numObjEqs);
	_CalcGrad(xNow, gPrev, gradEqs);
	
	bool exit_con = false;
	while(!exit_con)
	{
		for (int i = 0; i < n; i++) {
			H(i, i) = 1.0;
			p[i] = -gPrev[i];
		}
		k = 0;
		_VecNorm(p);

		while(true)
		{
			if (itCounter++ > itMax)
			{
				exit_con = true;
				break;
			}

			xPrev = xNow;
			_LinearSearch(xPrev, p, step, xNow, objEqs, numObjEqs);
			fNow = _CalcObj(xNow, objEqs, numObjEqs);
			std::cout << itCounter << " iterations, " <<	"f(x) = " << fNow << std::endl;

			if (fNow < eps)
			{
				exit_con = true;
				break;
			}

			_CalcGrad(xNow, gNow, gradEqs);

			if (_HTerminate(xPrev, xNow, fPrev, fNow, gNow))
			{
				exit_con = true;
				break;
			}

			if (fNow > fPrev) {
				_VecCopy(xNow, xPrev);
				break;
			}

			if (k == n) {
				fPrev = fNow;
				_VecCopy(gPrev, gNow);
				break;
			}
			checkCudaErrors(cudaMemcpy(d_gnow, gNow.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_gprev, gPrev.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_H, H.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_xnow, xNow.data(),	sizeof(double) * n, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_xprev, xPrev.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemset(d_sy, 0, sizeof(double)));

			void *kernelArgs[] = {
					(void*)&d_gprev, (void*)&d_gnow, (void*)&d_xprev, (void*)&d_xnow, (void*)&d_H, (void*)&d_p, (void*)&d_yTH, (void*)&d_Hy, (void*)&d_s, (void*)&d_y, (void*)&d_sy, (void*)&d_dot_result, (void*)&n
			};

			int sMemSize = sizeof(double) * ((THREADS_PER_BLOCK/32) + 1);
			int numBlocksPerSm = 0;
			int numThreads = THREADS_PER_BLOCK;

			checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&numBlocksPerSm, BFGSMultiply, numThreads, sMemSize));

			int numSms = deviceProp.multiProcessorCount;
			dim3 dimGrid(numSms * numBlocksPerSm, 1, 1),
				dimBlock(THREADS_PER_BLOCK, 1, 1);
			checkCudaErrors(cudaLaunchCooperativeKernel((void *)BFGSMultiply,
														dimGrid, dimBlock, kernelArgs,
														sMemSize, NULL));	

			checkCudaErrors(cudaDeviceSynchronize());

			_VecSub(gNow, gPrev, y);
			_VecSub(xNow, xPrev, s);
			double sy1 = _VecDot(s, y);

			std::vector<double> y1(n), s1(n);
			checkCudaErrors(cudaMemcpy(y1.data(), d_y, sizeof(double) * n, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(s1.data(), d_s, sizeof(double) * n, cudaMemcpyDeviceToHost));

			fPrev = fNow;
			gPrev = gNow;
			xPrev = xNow;
			double sy = 0;
			checkCudaErrors(cudaMemcpy(p.data(), d_p, sizeof(double) * n, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(&sy, d_sy, sizeof(double), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(H.data(), d_H, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

			

			if(sy < epsZero1)
			{
				exit_con = true;
				break;
			}
		}
	}

	std::cout << itCounter << " iterations" << std::endl;
	std::cout << "f(x) = " << fNow << std::endl;
	double dt = omp_get_wtime()-t0;
	printf("###Solver totally used %2.5f s ...\n", dt);

	//Put results back...
	if (fNow < eps) {
		printf("Solved!!!!\n");
		return true;
	}
	else {
		printf("Solver Failed!!!!\n");
		return false;
	}

	checkCudaErrors(cudaFree(d_gnow));
	checkCudaErrors(cudaFree(d_gprev));
	checkCudaErrors(cudaFree(d_xnow));
	checkCudaErrors(cudaFree(d_xprev));
	checkCudaErrors(cudaFree(d_s));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_yTH));
	checkCudaErrors(cudaFree(d_Hy));
	checkCudaErrors(cudaFree(d_p));
	checkCudaErrors(cudaFree(d_H));
	checkCudaErrors(cudaFree(d_sy));
	checkCudaErrors(cudaFree(d_dot_result));
}

int main()
{
	BFGSSolveEqs();
}

