#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

template <typename T>
T product(T* begin, T* end) {
    return std::accumulate(begin, end, 1, std::multiplies<T>());
}

namespace utec::algebra {

    template <typename T, size_t Rank>
    class Tensor {
        std::array<size_t, Rank> m_shape;
        std::vector<T> m_data;

    public:
        Tensor(const Tensor<T, Rank>& other) = default;
        Tensor(Tensor<T, Rank>&& other) = default;

        Tensor(const std::array<size_t, Rank>& shape)
            : m_shape(shape),
              m_data(product(shape.begin(), shape.end())) {}

        template <typename... Dims>
        Tensor(Dims... dims)
            : Tensor(std::array<size_t, Rank>{dims...}) {}

        template <typename... Idxs>
        T& operator()(Idxs... idxs) {
            std::array<size_t, Rank> idxs_arr{idxs...};
            size_t real_index = 0;
            // size_t step = product(m_shape.begin(), m_shape.end());
            //
            // for (size_t i = 0; i < Rank; i++) {
            //     step /= m_shape[Rank - 1 - i];
            //     real_index += step * idxs_arr[i];
            // }

            return m_data[real_index];
        }

        // template <typename... Idxs>
        // const T& operator()(Idxs... idxs) const {}

        const std::array<std::size_t, Rank>& shape() const noexcept {
            return m_shape;
        }

        void reshape(const std::array<std::size_t, Rank>& new_shape) {
            // if (product(new_shape.begin(), new_shape.end()) != m_data.size())
            // {
            //     throw std::invalid_argument(
            //         "Tensor::reshape() must keep the total number of
            //         elements");
            // }

            m_shape = new_shape;
        }

        template <typename... Dims>
        void reshape(Dims... dims) {
            reshape(std::array<size_t, Rank>{dims...});
        }

        void fill(const T& value) noexcept {
            std::fill(m_data.begin(), m_data.end(), value);
        }

        Tensor<T, Rank> operator+(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result;
            result.m_data = this->m_data;
            result.m_shape = this->m_shape;

            for (size_t i = 0; i < result.m_data.size(); i++) {
                result.m_data[i] = result.m_data[i] + other.m_data[i];
            }

            return result;
        }

        Tensor<T, Rank> operator-(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result;
            result.m_data = this->m_data;
            result.m_shape = this->m_shape;

            for (size_t i = 0; i < result.m_data.size(); i++) {
                result.m_data[i] = result.m_data[i] - other.m_data[i];
            }

            return result;
        }

        Tensor<T, Rank> operator*(const T& scalar) const {
            Tensor<T, Rank> result;
            result.m_data = this->m_data;
            result.m_shape = this->m_shape;

            for (auto& value : result.m_data) {
                value = value * scalar;
            }

            return result;
        }

        T& operator[](const size_t index) {
            return m_data[index];
        }

        const T& operator[](const size_t index) const {
            return m_data[index];
        }
    };

    // template <typename T>
    // class Tensor<T, 2> {
    //     Tensor transpose_2d() const;
    // };

}

#endif
