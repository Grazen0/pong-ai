#ifndef INCLUDE_UTEC_NEURAL_NETWORK_LOSS_H
#define INCLUDE_UTEC_NEURAL_NETWORK_LOSS_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "utec/algebra/tensor.h"

namespace utec::neural_network {
    using algebra::Tensor;
    template <typename T>
    class MSELoss {
        Tensor<T, 2> last_pred;
        Tensor<T, 2> last_target;

    public:
        auto forward(const Tensor<T, 2>& pred, const Tensor<T, 2>& target) -> T {
            if (pred.shape() != target.shape()) {
                throw std::invalid_argument("pred and target have incompatible shapes");
            }

            last_pred = pred;
            last_target = target;

            const std::size_t n = pred.shape()[0] * pred.shape()[1];
            T sum = T{0};

            for (std::size_t i = 0; i < n; ++i) {
                sum += std::pow(pred[i] - target[i], 2);
            }

            return sum / n;
        }

        auto backward() -> Tensor<T, 2> {
            const std::size_t n = last_pred.shape()[0] * last_pred.shape()[1];
            return (last_pred - last_target) * (2 / n);
        }
    };
}  // namespace utec::neural_network

#endif
