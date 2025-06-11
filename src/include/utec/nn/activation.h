#ifndef INCLUDE_UTEC_NEURAL_NETWORK_ACTIVATION_H
#define INCLUDE_UTEC_NEURAL_NETWORK_ACTIVATION_H

#include <cstddef>
#include "layer.h"
#include "utec/algebra/tensor.h"

namespace utec::neural_network {
    using algebra::Tensor;

    template <typename T>
    class ReLU : public ILayer<T> {
        Tensor<T, 2> mask;

    public:
        auto forward(const Tensor<T, 2>& x) -> Tensor<T, 2> override {
            const std::size_t size = x.shape()[0] * x.shape()[1];
            mask = Tensor<T, 2>{x.shape()};

            for (std::size_t i = 0; i < size; ++i) {
                mask[i] = x[i] > 0;
            }

            return x * mask;
        }

        auto backward(const Tensor<T, 2>& grad) -> Tensor<T, 2> override {
            return grad * mask;
        }
    };
}  // namespace utec::neural_network

#endif
