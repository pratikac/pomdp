#ifndef __linalg_h__
#define __linalg_h__

#include <unistd.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/MatrixFunctions>
using namespace Eigen;

typedef VectorXd vec;
typedef MatrixXd mat;
typedef MatrixXf matf;
typedef MatrixXi mati;
typedef VectorXf vecf;
typedef SparseMatrix<double> spmat;
typedef SparseVector<double> spvec;
typedef SparseMatrix<float> spmatf;
typedef SparseVector<float> spvecf;

typedef Triplet<float> triplet;

#endif
