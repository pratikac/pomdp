#ifndef __linalg_h__
#define __linalg_h__

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/MatrixFunctions>
using namespace Eigen;

typedef MatrixXf mat;
typedef MatrixXi mati;
typedef VectorXf vec;
typedef SparseMatrix<float> spmat;
typedef SparseVector<float> spvec;

typedef Triplet<float> triplet;

#define One(x)          vec::Constant(x,1,1)
#define Zero(x)         vec::Zero(x)
#define Identity(x,y)   mat::Identity(x,y)

#endif
