#ifndef INCLUDE_UTEC_ALGEBRA_TENSOR_H
#define INCLUDE_UTEC_ALGEBRA_TENSOR_H

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

template <typename Container>
auto product(const Container& cnt) {
    return std::accumulate(cnt.begin(), cnt.end(), 1,
                           std::multiplies<typename Container::value_type>());
}

namespace utec::algebra {
    template <typename T, std::size_t Rank>
    class Tensor {
        std::array<std::size_t, Rank> m_shape;
        std::array<std::size_t, Rank> m_steps;
        std::vector<T> m_data;

        void update_steps() {
            std::size_t step = 1;

            for (std::size_t i = Rank - 1; i != (std::size_t)-1; i--) {
                m_steps[i] = step;
                step *= m_shape[i];
            }
        }

    public:
        Tensor(const std::array<std::size_t, Rank>& shape)
            : m_shape(shape),
              m_data(product(shape)) {
            update_steps();
        }

        template <typename... Dims>
        Tensor(const Dims... dims)
            requires(sizeof...(Dims) == Rank)
            : m_shape{static_cast<std::size_t>(dims)...},
              m_data(product(m_shape)) {
            update_steps();
        }

        template <typename... Idxs>
        T& operator()(const Idxs... idxs)
            requires(sizeof...(Idxs) == Rank)
        {
            std::size_t i = 0;
            ((static_cast<std::size_t>(idxs) < m_shape[i++]
                  ? void()
                  : throw std::out_of_range("Tensor index out of bounds")),
             ...);

            i = 0;
            std::size_t physical_index = 0;
            ((physical_index += m_steps[i++] * static_cast<std::size_t>(idxs)),
             ...);

            return m_data[physical_index];
        }

        template <typename... Idxs>
        const T& operator()(const Idxs... idxs) const
            requires(sizeof...(Idxs) == Rank)
        {
            std::size_t i = 0;
            ((static_cast<std::size_t>(idxs) < m_shape[i++]
                  ? void()
                  : throw std::out_of_range("Tensor index out of bounds")),
             ...);

            i = 0;
            std::size_t physical_index = 0;
            ((physical_index += m_steps[i++] * static_cast<std::size_t>(idxs)),
             ...);

            return m_data[physical_index];
        }

        const std::array<std::size_t, Rank>& shape() const noexcept {
            return m_shape;
        }

        void reshape(const std::array<std::size_t, Rank>& new_shape) {
            const std::size_t new_size =
                product(new_shape.begin(), new_shape.end());

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
            const std::size_t new_size = (1 * ... * dims);

            if (new_size != m_data.size()) {
                throw std::invalid_argument(
                    "Tensor::reshape() must keep the total number of elements");
            }

            m_shape = {static_cast<std::size_t>(dims)...};
            update_steps();
        }

        void fill(const T& value) noexcept {
            std::fill(m_data.begin(), m_data.end(), value);
        }

        Tensor<T, Rank> operator+(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result{*this};

            for (std::size_t i = 0; i < result.m_data.size(); i++) {
                result.m_data[i] = result.m_data[i] + other.m_data[i];
            }

            return result;
        }

        Tensor<T, Rank> operator-(const Tensor<T, Rank>& other) const {
            Tensor<T, Rank> result{*this};

            for (std::size_t i = 0; i < result.m_data.size(); i++) {
                result.m_data[i] = result.m_data[i] - other.m_data[i];
            }

            return result;
        }

        Tensor<T, Rank> operator*(const T& scalar) const {
            Tensor<T, Rank> result(m_shape);

            for (std::size_t i = 0; i < m_data.size(); i++) {
                result.m_data[i] = m_data[i] * scalar;
            }

            return result;
        }

        Tensor<T, Rank> operator*(const Tensor<T, Rank>& other) const {
            if (m_shape == other.m_shape) {
                // Element-wise multiplication
                Tensor<T, Rank> result(m_shape);
                for (std::size_t i = 0; i < m_data.size(); i++) {
                    result.m_data[i] = m_data[i] * other.m_shape[i];
                }
                return result;
            }

            const bool is_this_smol =
                std::ranges::find(m_shape, 1) != m_shape.end();
            const bool is_other_smol =
                std::ranges::find(other.m_shape, 1) != other.m_shape.end();

            if (!is_this_smol && !is_other_smol) {
                throw std::invalid_argument("Tensor dimensions do not match");
            }

            // Broadcast
            const Tensor<T, Rank>& smol =
                (is_this_smol && !is_other_smol) ? *this : other;
            const Tensor<T, Rank>& big =
                (is_this_smol && !is_other_smol) ? other : *this;

            Tensor<T, Rank> result(big.m_shape);

            std::array<std::size_t, Rank> big_idx{};
            bool done = false;

            while (!done) {
                std::array<std::size_t, Rank> smol_idx{big_idx};

                for (std::size_t j = 0; j < Rank; j++) {
                    smol_idx[j] %= smol.m_shape[j];
                }

                std::apply(result, big_idx) =
                    std::apply(big, big_idx) * std::apply(smol, smol_idx);

                // Increment big idx with carries
                big_idx[0]++;
                for (std::size_t j = 0; j < Rank; j++) {
                    if (big_idx[j] < big.m_shape[j]) {
                        // Carry stops here
                        break;
                    }

                    big_idx[j] = 0;

                    if (j < Rank - 1) {
                        big_idx[j + 1]++;
                    } else {
                        done = true;
                    }
                }
            }

            return result;
        }

        T& operator[](const std::size_t index) {
            return m_data[index];
        }

        const T& operator[](const std::size_t index) const {
            return m_data[index];
        }

        bool operator==(const Tensor<T, Rank>& other) const {
            return m_shape == other.m_shape && m_data == other.m_data;
        }

        bool operator!=(const Tensor<T, Rank>& other) const {
            return !(*this == other);
        }

        Tensor<T, Rank> transpose_2d() const
            requires(Rank == 2)
        {
            Tensor<T, Rank> result(m_shape[1], m_shape[0]);

            for (std::size_t i = 0; i < m_shape[0]; i++) {
                for (std::size_t j = 0; j < m_shape[1]; j++) {
                    result(j, i) = (*this)(i, j);
                }
            }

            return result;
        }
    };
}

#endif
