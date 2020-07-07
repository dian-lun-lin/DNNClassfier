#pragma once
#include <DNN/headerdef.hpp>
#include <DNN/activation.hpp>
#include <DNN/optimizer.hpp>

struct DenseLayer {
  
  public:
  
    DenseLayer(size_t input_dim, size_t middle_dim, size_t output_dim, std::optional<Activation> a = {});

    Eigen::MatrixXf infer(const Eigen::MatrixXf& input) const;

    void update(Optimizer& optimizer, const float lrate);

    void forward_pass(const Eigen::MatrixXf& input);

    void backward_pass(const Eigen::MatrixXf& input, Eigen::MatrixXf& dz);

    size_t get_output_dim() const;

    std::optional<Activation> activation;

    size_t layer;

    Eigen::MatrixXf x; 
    Eigen::MatrixXf w; 
    Eigen::MatrixXf b; 

    Eigen::MatrixXf dw; 
    Eigen::MatrixXf db; 

};

//-----------------------------------------------------------------------------
//Definition of DenseLayer
//-----------------------------------------------------------------------------

DenseLayer::DenseLayer(size_t input_dim, size_t output_dim, size_t layer, std::optional<Activation> a): 
  w{Eigen::MatrixXf::Random(input_dim, output_dim)},
  b{Eigen::MatrixXf::Random(1, output_dim)},
  layer{layer},
  activation{std::move(a)}
{
  dw.resize(input_dim, output_dim);
  db.resize(1, output_dim);
}

Eigen::MatrixXf DenseLayer::infer(const Eigen::MatrixXf& input) const {
  Eigen::MatrixXf output = input * w + b.replicate(input.rows(), 1); 

  if(activation) {
    activate(output, *activation);
  }
  return output;
}

void DenseLayer::forward_pass(const Eigen::MatrixXf& input) {
  x = input * w + b.replicate(input.rows(), 1); 

  if(activation) {
    activate(x, *activation);
  }
}

void DenseLayer::backward_pass(const Eigen::MatrixXf& input, Eigen::MatrixXf& dz) {
  //last layer dz = dloss;
  //
  if(activation) {
    dz = dz.cwiseProduct(deactivate(x, *activation));
  }
  
  dw = input.transpose() * dz;

  db = dz.colwise().sum();

    
  //not first layer
  if(layer) {
    dz = dz * w.transpose();
  }

}

size_t DenseLayer::get_output_dim() const {
  return x.cols();
}

void DenseLayer::update(Optimizer& optimizer, const float lrate) {
  std::visit([this, &lrate](auto&& opt){
    opt.set_learning_rate(lrate);
    opt.update(w, dw);
    opt.update(b, db);
  }, optimizer);
}

//variant-----------------------------------------
using Layer = std::variant<DenseLayer>;
