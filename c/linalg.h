#ifndef __linalg_h__
#define __linalg_h__

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/MatrixFunctions>
using namespace Eigen;

typedef MatrixXd mat;
typedef MatrixXf matf;
typedef MatrixXi mati;
typedef VectorXf vecf;
typedef VectorXd vec;
typedef SparseMatrix<float> spmat;
typedef SparseVector<float> spvec;

typedef Triplet<float> triplet;

#endif
