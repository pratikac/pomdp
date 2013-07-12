#ifndef __linalg_h__
#define __linalg_h__

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/MatrixFunctions>
using namespace Eigen;

typedef MatrixXf mat;
typedef MatrixXi mati;
typedef VectorXf vec;
typedef VectorXd vecd;
typedef SparseMatrix<float> spmat;
typedef SparseVector<float> spvec;

typedef Triplet<float> triplet;

#endif
