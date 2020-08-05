#pragma once
#include <DNN/headerdef.hpp>
#include <DNN/layer.hpp>
#include <DNN/loss.hpp>
#include <DNN/optimizer.hpp>
#include <DNN/activation.hpp>

class DNNClassfier {
  
  public:

    void dense_layer(size_t input_dim, size_t output_dim, const Activation& a = {});

    template<typename C>
    void train(Eigen::MatrixXf& inputs, Eigen::VectorXi& labels, const size_t epoch, const size_t batch_size, const float lrate, C&& callable);

    Eigen::VectorXi infer(const Eigen::MatrixXf& inputs) const;

  private:

    std::vector<Layer> _layers;

    Optimizer _opt{std::in_place_type<GradientDescent>};
    
    Loss _loss{std::in_place_type<SoftmaxCrossEntropy>};

    void _shuffle(Eigen::MatrixXf& r, Eigen::VectorXi& l);

    void _update(float lrate);

    void _optimize(const Eigen::MatrixXf& inputs, const Eigen::VectorXi labels, const float lrate);
      
};

//--------------------------------------------------------------------
//Definition of DNN
//--------------------------------------------------------------------

void DNNClassfier::dense_layer(size_t input_dim, size_t output_dim, const Activation& a) {

  _layers.emplace_back(std::in_place_type<DenseLayer>, input_dim, output_dim, _layers.size(), a);
}

Eigen::VectorXi DNNClassfier::infer(const Eigen::MatrixXf& inputs) const {
  Eigen::MatrixXf res = inputs;
  for(auto& layer: _layers) {
    res = std::visit([&](auto&& l){ return l.infer(res); }, layer);
  }

  //softmax 

  res = (res - res.rowwise().maxCoeff().replicate(1, res.cols())).array().exp().matrix();
  res = res.cwiseQuotient(res.rowwise().sum().replicate(1, res.cols()));

  Eigen::VectorXi ans{inputs.rows()};

  
  for(int i = 0; i < inputs.rows(); ++i) {
    res.row(i).maxCoeff(&ans(i));
  }
  

  return ans;
}

template <typename C>
void DNNClassfier::train(Eigen::MatrixXf& inputs, Eigen::VectorXi& labels, const size_t epoch, const size_t batch_size, const float lrate, C&& callable) {

  for(size_t i = 0; i < epoch; ++i) {
    _shuffle(inputs, labels);

    for(size_t n = 0; n < inputs.rows(); n += batch_size) {
      size_t num_data = ((n + batch_size) > inputs.rows()) ? inputs.rows() - n : batch_size;
      _optimize(inputs.middleRows(n, num_data), labels.middleRows(n, num_data), lrate);
    }

    if constexpr(std::is_invocable_v<C, DNNClassfier&>) {
      callable(*this);
    }
  }

}

void DNNClassfier::_optimize(const Eigen::MatrixXf& inputs, const Eigen::VectorXi labels, const float lrate) {
  //forwarding
  for(size_t j = 0; j < _layers.size(); ++j) {
    Eigen::MatrixXf x = (j == 0) ? inputs : std::visit([](auto&& layer){return layer.x; }, _layers[j - 1]);

    std::visit([&](auto&& layer){ layer.forward_pass(x); }, _layers[j]);
  }

  auto output = std::visit([&](auto&& layer){ return (layer.x); }, _layers.back());

  //softmax
  output = (output - output.rowwise().maxCoeff().replicate(1, output.cols())).array().exp().matrix();
  output = output.cwiseQuotient(output.rowwise().sum().replicate(1, output.cols()));

  //get softmax loss
  Eigen::MatrixXf dz = std::visit([&](auto&& loss){ return loss.dloss(output, labels); }, _loss);

  //backwarding
  for(int k = _layers.size() - 1; k >= 0; --k) {
    auto x = (k == 0) ? inputs : std::visit([](auto&& layer){ return layer.x; }, _layers[k - 1]);

    std::visit([&](auto&& layer){ layer.backward_pass(x, dz); }, _layers[k]);
  }

  _update(lrate / std::sqrt(inputs.rows()));
}

void DNNClassfier::_update(float lrate) {
  for(auto& l: _layers) {
    std::visit([&](auto&& layer){ 
      layer.update(_opt, lrate);
    }, l);
  }
}

void DNNClassfier::_shuffle(Eigen::MatrixXf& r, Eigen::VectorXi& l) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm{r.rows()}; 
  perm.setIdentity();
  std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), std::default_random_engine(seed));
  r = perm * r;
  l = perm * l;
}
