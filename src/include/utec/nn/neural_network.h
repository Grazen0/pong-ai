#ifndef INCLUDE_UTEC_NN_NEURAL_NETWORK_H
#define INCLUDE_UTEC_NN_NEURAL_NETWORK_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>
#include "layer.h"
#include "loss.h"

namespace utec::neural_network {
    using algebra::Tensor;

    template <typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        MSELoss<T> criterion;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers.emplace_back(std::move(layer));
        }

        auto forward(Tensor<T, 2> x) -> Tensor<T, 2> {
            return std::ranges::fold_left(layers, std::move(x), [](auto x, const auto& layer) {
                return layer->forward(std::move(x));
            });
        }

        void backward(Tensor<T, 2> grad) {
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backward(grad);
            }
        }

        void optimize(const T learning_rate) {}

        auto train(const Tensor<T, 2>& x,
                   const Tensor<T, 2>& y,
                   const std::size_t epochs,
                   const T learning_rate) -> T {
            if (x.shape()[0] != y.shape()[0]) {
                throw std::invalid_argument("x and y have incompatible shapes");
            }

            for (std::size_t ep = 0; ep < epochs; ++ep) {
                for (std::size_t i = 0; i < x.shape()[0]; ++i) {
                    Tensor<T, 2> x_i(x.shape()[1], 1);
                    Tensor<T, 2> target(y.shape()[1], 1);

                    for (std::size_t j = 0; j < x_i.shape()[0]; ++j) {
                        x_i(j, 0) = x(i, j);
                    }

                    for (std::size_t j = 0; j < target.shape()[0]; ++j) {
                        target(j, 0) = y(i, j);
                    }

                    const Tensor<T, 2> pred = forward(x_i);
                    const T loss = criterion.forward(pred, target);
                    const Tensor<T, 2> grad = criterion.backward();
                    backward(grad * loss);
                }
            }

            return T{0};
        }
    };
}  // namespace utec::neural_network

#endif
