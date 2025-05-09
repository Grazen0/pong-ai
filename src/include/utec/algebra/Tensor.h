#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename T>
T product(T* begin, T* end) {
    return std::accumulate(begin, end, 1, std::multiplies<T>());
}

namespace utec::algebra {
    template <typename T, size_t Rank>
    class Tensor {
        std::array<size_t, Rank> m_shape;
        std::array<size_t, Rank> m_steps;
        std::vector<T> m_data;

        void update_steps() {
            size_t step = 1;

            for (size_t i = Rank - 1; i != (size_t)-1; i--) {
                m_steps[i] = step;
                step *= m_shape[i];
            }
        }

    public:
        Tensor(const Tensor<T, Rank>& other) = default;
        Tensor(Tensor<T, Rank>&& other) = default;

        Tensor(const std::array<size_t, Rank>& shape)
            : m_shape(shape),
              m_data(product(shape.begin(), shape.end())) {
            update_steps();
        }

        template <typename... Dims>
        Tensor(Dims... dims)
            requires(sizeof...(Dims) == Rank)
            : Tensor(std::array<size_t, Rank>{static_cast<size_t>(dims)...}) {
            update_steps();
        }

        template <typename... Idxs>
        T& operator()(Idxs... idxs)
            requires(sizeof...(Idxs) == Rank)
        {
            size_t physical_index = 0;
            size_t i = 0;
            ((physical_index += m_steps[i++] * idxs), ...);
            return m_data[physical_index];
        }

        template <typename... Idxs>
        const T& operator()(Idxs... idxs) const
            requires(sizeof...(Idxs) == Rank)
        {
            size_t physical_index = 0;
            size_t i = 0;
            ((physical_index += m_steps[i++] * idxs), ...);
            return m_data[physical_index];
        }

        const std::array<std::size_t, Rank>& shape() const noexcept {
            return m_shape;
        }

        void reshape(const std::array<std::size_t, Rank>& new_shape) {
            const size_t new_size = product(new_shape.begin(), new_shape.end());

            if (new_size != m_data.size()) {
                throw std::invalid_argument(
                    "Tensor::reshape() must keep the total number of elements");
            }

            m_shape = new_shape;
            update_steps();
        }

        template <typename... Dims>
        void reshape(Dims... dims)
            requires(sizeof...(Dims) == Rank)
        {
            const size_t new_size = (1 * ... * dims);

            if (new_size != m_data.size()) {
                throw std::invalid_argument(
                    "Tensor::reshape() must keep the total number of elements");
            }

            m_shape = {static_cast<size_t>(dims)...};
            update_steps();
        }

        void fill(const T& value) noexcept {
            std::fill(m_data.begin(), m_data.end(), value);
        }

        Tensor<T, Rank> operator+(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result{*this};

            for (size_t i = 0; i < result.m_data.size(); i++) {
                result.m_data[i] = result.m_data[i] + other.m_data[i];
            }

            return result;
        }

        Tensor<T, Rank> operator-(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result{*this};

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

        // Tensor<T, Rank> transpose_2d()
        //     requires(Rank == 2)
        // {}
    };
}

#endif
