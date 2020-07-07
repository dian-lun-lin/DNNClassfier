#include <DNN/dnn.hpp>
#include <DNN/utility.hpp>

int main() {

  Eigen::MatrixXf images = read_mnist_image<Eigen::MatrixXf>("train-images-idx3-ubyte") / 255.0;
  Eigen::VectorXi labels = read_mnist_label<Eigen::VectorXi>("train-labels-idx1-ubyte");

  DNNClassfier dnn;
  dnn.dense_layer(images.cols(), 60, Activation::ReLU);
  dnn.dense_layer(60, 30, Activation::ReLU);
  dnn.dense_layer(30, 10);

  dnn.train(images, labels, 10, 64, 0.01, [&, i=0](auto&& dnn) mutable {
    auto c = ((dnn.infer(images) - labels).array() == 0).count();
    auto t = images.rows();
    std::cout << "Epoch: " << i++ << "accuracy = " << c/float(t) << std::flush;
  });
}
