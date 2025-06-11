#ifndef INCLUDE_UTEC_NEURAL_NETWORK_ILAYER_H
#define INCLUDE_UTEC_NEURAL_NETWORK_ILAYER_H

#include "utec/algebra/tensor.h"

namespace utec::neural_network {

    template <typename T>
    class ILayer {
    public:
        virtual ~ILayer() = default;

        virtual auto forward(const algebra::Tensor<T, 2>& x) -> algebra::Tensor<T, 2> = 0;

        virtual auto backward(const algebra::Tensor<T, 2>& grad) -> algebra::Tensor<T, 2> = 0;
    };
}  // namespace utec::neural_network

#endif
