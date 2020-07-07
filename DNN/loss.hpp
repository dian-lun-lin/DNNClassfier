#pragma once
#include <DNN/headerdef.hpp>

class SoftmaxCrossEntropy {

  public:

    Eigen::MatrixXf dloss(const Eigen::MatrixXf& x, const Eigen::VectorXi& labels);

    //Eigen::MatrixXf loss(const Eigen::MatrixXf& x);
};

//---------------------------------------------------------------------------------
//Definition of SoftmaxCrossEntropy
//---------------------------------------------------------------------------------

//Eigen::MatrixXf SoftmaxCrossEntropy::loss(const Eigen::MatrixXf& x) {
  //Eigen::MatrixXf res = x.exp() / (x.exp().sum());
  //return res;
//}

inline
Eigen::MatrixXf SoftmaxCrossEntropy::dloss(const Eigen::MatrixXf& x, const Eigen::VectorXi& labels) {
  assert(x.rows() == labels.rows() && labels.cols() == 1);

  //return x - labels.replicate(1, x.cols());

  Eigen::MatrixXf res = x;

  for(int i = 0; i < x.rows(); ++i) {
    res(i, labels(i)) -= 1.0f;
  }

  return res;
}

//variant-----------------------------------------
using Loss = std::variant<SoftmaxCrossEntropy>;

