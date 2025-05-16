#ifndef INCLUDE_UTEC_ALGEBRA_TENSOR_H
#define INCLUDE_UTEC_ALGEBRA_TENSOR_H

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

template <typename T>
T product(T* begin, T* end) {
    return std::accumulate(begin, end, 1, std::multiplies<T>());
}

namespace utec::algebra {
    template <typename T, std::size_t Rank>
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
        Tensor(const std::array<size_t, Rank>& shape)
            : m_shape(shape),
              m_data(product(shape.begin(), shape.end())) {
            update_steps();
        }

        template <typename... Dims>
        Tensor(const Dims... dims)
            requires(sizeof...(Dims) == Rank)
            : m_shape{static_cast<size_t>(dims)...},
              m_data(product(m_shape.begin(), m_shape.end())) {
            update_steps();
        }

        template <typename... Idxs>
        T& operator()(const Idxs... idxs)
            requires(sizeof...(Idxs) == Rank)
        {
            size_t i = 0;
            const size_t physical_index =
                (0 + ... +
                 (static_cast<size_t>(idxs) < m_shape[i++]
                      ? m_steps[i] * static_cast<size_t>(idxs)
                      : throw std::out_of_range("Tensor index out of bounds")));

            return m_data[physical_index];
        }

        template <typename... Idxs>
        const T& operator()(const Idxs... idxs) const
            requires(sizeof...(Idxs) == Rank)
        {
            size_t i = 0;
            const size_t physical_index =
                (0 + ... +
                 (static_cast<size_t>(idxs) < m_shape[i++]
                      ? m_steps[i] * static_cast<size_t>(idxs)
                      : throw std::out_of_range("Tensor index out of bounds")));

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
        void reshape(const Dims... dims)
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
            Tensor<T, Rank> result(m_shape);

            for (size_t i = 0; i < m_data.size(); i++) {
                result.m_data[i] = m_data[i] * scalar;
            }

            return result;
        }

        T& operator[](const size_t index) {
            return m_data[index];
        }

        const T& operator[](const size_t index) const {
            return m_data[index];
        }

        Tensor<T, Rank> transpose_2d() const
            requires(Rank == 2)
        {
            Tensor<T, Rank> result(m_shape[1], m_shape[0]);

            for (size_t i = 0; i < m_shape[0]; i++) {
                for (size_t j = 0; j < m_shape[1]; j++) {
                    result(j, i) = (*this)(i, j);
                }
            }

            return result;
        }
    };
}

#endif
