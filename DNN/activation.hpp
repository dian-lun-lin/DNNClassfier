#pragma once
#include <DNN/headerdef.hpp>

enum Activation {
  ReLU
};

void RELU(Eigen::MatrixXf& x);

Eigen::MatrixXf dRELU(const Eigen::MatrixXf& x);

void activate(Activation& act);

Eigen::MatrixXf deactivate(const Eigen::MatrixXf& x, const Activation& a);


//---------------------------------------------------
//Definition of activation functions
//---------------------------------------------------

inline
void activate(Eigen::MatrixXf& x, const Activation& a) {
  switch(a) {
    case ReLU:
      RELU(x);
      break;
    default:
      throw std::runtime_error("Not an activation function");
  }
}

inline
Eigen::MatrixXf deactivate(const Eigen::MatrixXf& x, const Activation& a) {
  switch(a) {
    case ReLU:
      return dRELU(x);
    default:
      throw std::runtime_error("Not an activation function");
  }
}

inline 
void RELU(Eigen::MatrixXf& x) {
  for(int i = 0; i < x.rows(); ++i) {
    for(int j = 0; j < x.cols(); ++j) {
      if(x(i, j) <= 0.0f) {
        x(i, j) = 0.0f;
      }
    }
  }
}

inline
Eigen::MatrixXf dRELU(const Eigen::MatrixXf& x) {
  Eigen::MatrixXf res{x.rows(), x.cols()};
  for(int i = 0; i < x.rows(); ++i) {
    for(int j = 0; j < x.cols(); ++j) {
      res(i, j) = (x(i, j) > 0.0f) ? 1.0f : 0.0f;
    }
  }
  return res;
}

