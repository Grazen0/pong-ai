#ifndef INCLUDE_UTEC_NEURAL_NETWORK_DENSE_H
#define INCLUDE_UTEC_NEURAL_NETWORK_DENSE_H

#include <cstddef>
#include <stdexcept>
#include "layer.h"
#include "utec/algebra/tensor.h"

namespace utec::neural_network {
    using algebra::Tensor;

    template <typename T>
    class Dense : public ILayer<T> {
        Tensor<T, 2> w;
        Tensor<T, 2> dw;
        Tensor<T, 1> b;
        Tensor<T, 1> db;
        Tensor<T, 2> last_x;

    public:
        Dense(const std::size_t in_feats, const std::size_t out_feats)
            : w(out_feats, in_feats),
              dw(out_feats, in_feats),
              b(out_feats),
              db(out_feats) {}

        auto forward(const Tensor<T, 2>& x) -> Tensor<T, 2> override {
            if (x.shape()[0] != w.shape()[1] || x.shape()[1] != 1) {
                throw std::invalid_argument("invalid x shape");
            }

            last_x = x;
            Tensor<T, 2> out(w.shape()[0], 1);

            for (std::size_t i = 0; i < out.shape()[0]; ++i) {
                for (std::size_t j = 0; j < x.shape()[0]; ++j) {
                    out(i, 0) += x(j, 0) * w(i, j);
                }

                out(i, 0) += b(i);
            }

            return out;
        }

        auto backward(const Tensor<T, 2>& grad) -> Tensor<T, 2> override {
            if (grad.shape()[0] != w.shape()[0] || grad.shape()[1] != 1) {
                throw std::invalid_argument("invalid grad shape");
            }

            return grad;
        }
    };
}  // namespace utec::neural_network

#endif
