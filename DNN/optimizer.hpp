#pragma once

#include <DNN/layer.hpp>
#include <DNN/headerdef.hpp>

class GradientDescent {
  
  public:

    void update(Eigen::MatrixXf& x, const Eigen::MatrixXf& dx) const;

    void set_learning_rate(float lrate);
    
  private:
  
    float _learning_rate{0.01f};
};

//---------------------------------------------------------------
//Definition of GradientDescent
//---------------------------------------------------------------

inline
void GradientDescent::set_learning_rate(float lrate) {
  _learning_rate = lrate;
}

inline
void GradientDescent::update(Eigen::MatrixXf& x, const Eigen::MatrixXf& dx) const {
  x -= _learning_rate * dx;
}


//variant-----------------------------------------------------------
using Optimizer = std::variant<GradientDescent>;
