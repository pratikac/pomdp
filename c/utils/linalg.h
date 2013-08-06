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
typedef SparseMatrix<float> spmat;
typedef SparseVector<float> spvec;

typedef Triplet<float> triplet;

#endif
