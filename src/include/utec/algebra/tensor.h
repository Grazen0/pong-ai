#ifndef INCLUDE_UTEC_ALGEBRA_TENSOR_H
#define INCLUDE_UTEC_ALGEBRA_TENSOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
    template <std::size_t Size>
    constexpr void apply_with_counter(const auto fn, const std::array<std::size_t, Size>& size) {
        std::array<std::size_t, Size> index{};

        const std::size_t total_size = std::ranges::fold_left(size, 1, std::multiplies<>{});

        for (std::size_t i = 0; i < total_size; ++i) {
            fn(index);

            index[0]++;

            for (std::size_t j = 0; j < Size; ++j) {
                if (index[j] < size[j]) {
                    // Carry stops here
                    break;
                }

                index[j] = 0;
                if (j < Size - 1) {
                    index[j + 1]++;
                }
            }
        }
    }
}  // namespace

namespace utec::algebra {
    template <typename T, std::size_t Rank>
    class Tensor {
        std::array<std::size_t, Rank> m_shape;
        std::array<std::size_t, Rank> m_steps;
        std::vector<T> m_data;

        constexpr void update_steps() {
            std::size_t step = 1;

            for (std::size_t i = Rank - 1; i != static_cast<std::size_t>(-1); i--) {
                m_steps[i] = step;
                step *= m_shape[i];
            }
        }

        template <std::size_t Rank2>
            requires(Rank2 > Rank)
        constexpr auto broadcast(const Tensor<T, Rank2>& rhs, const auto fn) const
            -> Tensor<T, Rank2> {
            std::array<std::size_t, Rank2> new_shape;

            std::ranges::copy(m_shape, new_shape.begin());
            std::fill(new_shape.begin() + Rank, new_shape.end(), 1);

            Tensor<T, Rank2> lhs_expanded{new_shape};
            std::ranges::copy(m_data, lhs_expanded.m_data.begin());

            return lhs_expanded.broadcast(rhs, fn);
        }

        template <std::size_t Rank2>
            requires(Rank2 < Rank)
        constexpr auto broadcast(const Tensor<T, Rank2>& rhs, const auto fn) const
            -> Tensor<T, Rank> {
            std::array<std::size_t, Rank> new_shape;

            std::ranges::copy(rhs.m_shape, new_shape.begin());
            std::fill(new_shape.begin() + Rank2, new_shape.end(), 1);

            Tensor<T, Rank> rhs_expanded{new_shape};
            std::ranges::copy(rhs.m_data, rhs_expanded.m_data.begin());

            return broadcast(rhs_expanded, fn);
        }

        constexpr auto broadcast(const Tensor<T, Rank>& rhs, const auto fn) const
            -> Tensor<T, Rank> {
            if (m_shape == rhs.m_shape) {
                // Element-wise
                Tensor<T, Rank> result{m_shape};
                for (std::size_t i = 0; i < m_data.size(); ++i) {
                    result[i] = fn(m_data[i], rhs[i]);
                }
                return result;
            }

            std::array<std::size_t, Rank> result_shape;

            for (std::size_t i = 0; i < Rank; ++i) {
                if (m_shape[i] == rhs.m_shape[i]) {
                    result_shape[i] = m_shape[i];
                } else if (m_shape[i] == 1 || rhs.m_shape[i] == 1) {
                    result_shape[i] = std::max(m_shape[i], rhs.m_shape[i]);
                } else {
                    throw std::invalid_argument(
                        "Shapes do not match and they are not compatible for broadcasting");
                }
            }

            Tensor<T, Rank> result{result_shape};

            apply_with_counter(
                [&](const auto& result_index) {
                    std::array<std::size_t, Rank> lhs_index{result_index};
                    std::array<std::size_t, Rank> rhs_index{result_index};

                    for (std::size_t i = 0; i < Rank; ++i) {
                        lhs_index[i] %= m_shape[i];
                    }

                    for (std::size_t i = 0; i < Rank; ++i) {
                        rhs_index[i] %= rhs.m_shape[i];
                    }

                    std::apply(result, result_index) =
                        fn(std::apply(*this, lhs_index), std::apply(rhs, rhs_index));
                },
                result_shape);

            return result;
        }

        template <typename... Idxs>
            requires(sizeof...(Idxs) == Rank)
        constexpr auto physical_index(const Idxs... idxs) const -> std::size_t {
            const std::array<std::size_t, Rank> idxs_arr{static_cast<std::size_t>(idxs)...};

            std::size_t physical_index = 0;

            for (std::size_t i = 0; i < Rank; ++i) {
                if (idxs_arr[i] >= m_shape[i]) {
                    throw std::out_of_range("Tensor index out of bounds");
                }
                physical_index += m_steps[i] * idxs_arr[i];
            }

            return physical_index;
        }

    public:
        Tensor() = default;

        explicit Tensor(const std::array<std::size_t, Rank>& shape)
            : m_shape(shape),
              m_data(std::ranges::fold_left(shape, 1, std::multiplies<>{})) {
            update_steps();
        }

        template <typename... Dims>
            requires(sizeof...(Dims) == Rank)
        explicit Tensor(const Dims... dims)
            : m_shape({static_cast<std::size_t>(dims)...}),
              m_data((1 * ... * dims)) {
            update_steps();
        }

        [[nodiscard]] constexpr auto shape() const noexcept
            -> const std::array<std::size_t, Rank>& {
            return m_shape;
        }

        [[nodiscard]] constexpr auto size() const -> std::size_t {
            return m_data.size();
        }

        template <typename... Dims>
            requires(sizeof...(Dims) == Rank)
        constexpr void reshape(const Dims... dims) {
            m_shape = std::array{static_cast<std::size_t>(dims)...};
            m_data.resize((1 * ... * dims));
            update_steps();
        }

        constexpr void fill(const T& value) noexcept {
            std::ranges::fill(m_data, value);
        }

        [[nodiscard]] constexpr auto operator[](const std::size_t index) -> T& {
            return m_data.at(index);
        }

        [[nodiscard]] constexpr auto operator[](const std::size_t index) const -> const T& {
            return m_data.at(index);
        }

        [[nodiscard]] constexpr auto operator()(const auto... idxs) -> T& {
            return m_data[physical_index(idxs...)];
        }

        [[nodiscard]] constexpr auto operator()(const auto... idxs) const -> const T& {
            return m_data[physical_index(idxs...)];
        }

        template <std::ranges::sized_range Range>
            requires std::convertible_to<std::ranges::range_value_t<Range>, T>
        constexpr auto operator=(const Range& range) -> Tensor<T, Rank>& {
            if (std::ranges::size(range) != m_data.size()) {
                throw std::invalid_argument("Data size does not match tensor size");
            }

            std::ranges::copy(range, m_data.begin());
            return *this;
        }

        template <std::ranges::input_range Range>
            requires(!std::ranges::sized_range<Range> &&
                     std::convertible_to<std::ranges::range_value_t<Range>, T>)
        constexpr auto operator=(const Range& range) -> Tensor<T, Rank>& {
            if (std::ranges::distance(range) != m_data.size()) {
                throw std::invalid_argument("Data size does not match tensor size");
            }

            std::ranges::copy(range, m_data.begin());
            return *this;
        }

        constexpr auto operator=(const std::initializer_list<T>& list) -> Tensor<T, Rank>& {
            if (list.size() != m_data.size()) {
                throw std::invalid_argument("Data size does not match tensor size");
            }

            m_data = list;
            return *this;
        }

        template <std::size_t Rank2>
        [[nodiscard]] constexpr auto operator+(const Tensor<T, Rank2>& rhs) const
            -> Tensor<T, std::max(Rank, Rank2)> {
            return broadcast(rhs, std::plus<T>{});
        }

        [[nodiscard]] constexpr auto operator+(const T& value) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result{m_shape};
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& x) { return x + value; });
            return result;
        }

        template <std::size_t Rank2>
        [[nodiscard]] constexpr auto operator-(const Tensor<T, Rank2>& rhs) const
            -> Tensor<T, std::max(Rank, Rank2)> {
            return broadcast(rhs, std::minus<T>{});
        }

        [[nodiscard]] constexpr auto operator-(const T& value) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result{m_shape};
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& x) { return x - value; });
            return result;
        }

        [[nodiscard]] constexpr auto operator-() const -> Tensor<T, Rank> {
            Tensor<T, Rank> result{m_shape};
            std::ranges::transform(m_data, result.m_data.begin(), std::negate<T>{});
            return result;
        }

        template <std::size_t Rank2>
        [[nodiscard]] constexpr auto operator*(const Tensor<T, Rank2>& rhs) const
            -> Tensor<T, std::max(Rank, Rank2)> {
            return broadcast(rhs, std::multiplies<T>{});
        }

        [[nodiscard]] constexpr auto operator*(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result{m_shape};
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& x) { return x * scalar; });
            return result;
        }
        template <std::size_t Rank2>
        [[nodiscard]] constexpr auto operator/(const Tensor<T, Rank2>& rhs) const
            -> Tensor<T, std::max(Rank, Rank2)> {
            return broadcast(rhs, std::divides<T>{});
        }

        [[nodiscard]] constexpr auto operator/(const T& scalar) const -> Tensor<T, Rank> {
            Tensor<T, Rank> result{m_shape};
            std::ranges::transform(m_data, result.m_data.begin(),
                                   [&](const T& x) { return x / scalar; });
            return result;
        }

        [[nodiscard]] constexpr auto operator==(const Tensor<T, Rank>& other) const -> bool {
            return m_shape == other.m_shape && m_data == other.m_data;
        }

        [[nodiscard]] constexpr auto operator!=(const Tensor<T, Rank>& other) const -> bool {
            return !(*this == other);
        }

        [[nodiscard]] constexpr auto begin() noexcept {
            return m_data.begin();
        }

        [[nodiscard]] constexpr auto end() noexcept {
            return m_data.end();
        }

        [[nodiscard]] constexpr auto begin() const noexcept {
            return m_data.begin();
        }

        [[nodiscard]] constexpr auto end() const noexcept {
            return m_data.end();
        }

        [[nodiscard]] constexpr auto transpose_2d() -> Tensor<T, 2> {
            Tensor<T, 2> result(m_shape[1], m_shape[0]);

            for (std::size_t i = 0; i < m_shape[0]; ++i) {
                for (std::size_t j = 0; j < m_shape[1]; ++j) {
                    result(j, i) = (*this)(i, j);
                }
            }

            return result;
        }

        [[nodiscard]] constexpr auto transpose_2d() -> Tensor<T, Rank>
            requires(Rank > 2)
        {
            std::array<std::size_t, Rank> new_shape{m_shape};
            std::swap(new_shape[Rank - 2], new_shape[Rank - 1]);

            Tensor<T, Rank> result{new_shape};
            std::array<std::size_t, Rank - 2> size{};

            std::copy(m_shape.begin(), m_shape.end() - 2, size.begin());

            apply_with_counter(
                [&](const auto& index) {
                    std::array<std::size_t, Rank> full_index;
                    std::copy(index.begin(), index.end(), full_index.begin());

                    for (std::size_t i = 0; i < m_shape[Rank - 2]; ++i) {
                        for (std::size_t j = 0; j < m_shape[Rank - 1]; ++j) {
                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = j;
                            const T src = std::apply(*this, full_index);

                            full_index[Rank - 2] = j;
                            full_index[Rank - 1] = i;
                            T& dest = std::apply(result, full_index);

                            dest = src;
                        }
                    }
                },
                size);

            return result;
        }
    };

    template <typename T>
    [[nodiscard]] constexpr auto matrix_product(const Tensor<T, 2>& lhs, const Tensor<T, 2>& rhs)
        -> Tensor<T, 2> {
        if (lhs.shape()[1] != rhs.shape()[0]) {
            throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
        }

        // Simple matrix multiplication
        Tensor<T, 2> result(lhs.shape()[0], rhs.shape()[1]);

        for (std::size_t i = 0; i < result.shape()[0]; ++i) {
            for (std::size_t j = 0; j < result.shape()[1]; ++j) {
                for (std::size_t k = 0; k < lhs.shape()[1]; ++k) {
                    result(i, j) += lhs(i, k) * rhs(k, j);
                }
            }
        }

        return result;
    }

    template <typename T, std::size_t Rank>
        requires(Rank > 2)
    [[nodiscard]] constexpr auto matrix_product(const Tensor<T, Rank>& lhs,
                                                const Tensor<T, Rank>& rhs) -> Tensor<T, Rank> {
        for (std::size_t i = 0; i < Rank - 2; ++i) {
            if (lhs.shape()[i] != rhs.shape()[i]) {
                throw std::invalid_argument("Incompatible batch dimensions for multiplication");
            }
        }

        std::array<std::size_t, Rank> new_shape{lhs.shape()};
        new_shape[Rank - 1] = rhs.shape()[Rank - 1];

        Tensor<T, Rank> result(new_shape);
        std::array<std::size_t, Rank - 2> size{};
        std::copy(lhs.shape().begin(), lhs.shape().end() - 2, size.begin());

        apply_with_counter(
            [&](const auto& index) {
                std::array<std::size_t, Rank> full_index;
                std::ranges::copy(index, full_index.begin());

                for (std::size_t i = 0; i < result.shape()[Rank - 2]; ++i) {
                    for (std::size_t j = 0; j < result.shape()[Rank - 1]; ++j) {
                        for (std::size_t k = 0; k < lhs.shape()[Rank - 1]; ++k) {
                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = k;
                            const T& src1 = std::apply(lhs, full_index);

                            full_index[Rank - 2] = k;
                            full_index[Rank - 1] = j;
                            const T& src2 = std::apply(rhs, full_index);

                            full_index[Rank - 2] = i;
                            full_index[Rank - 1] = j;
                            T& dest = std::apply(result, full_index);

                            dest += src1 * src2;
                        }
                    }
                }
            },
            size);

        return result;
    }

}  // namespace utec::algebra

namespace {
    using utec::algebra::Tensor;

    template <typename T, std::size_t Rank>
    [[nodiscard]] constexpr auto operator+(const T& lhs_value, const Tensor<T, Rank>& tensor)
        -> Tensor<T, Rank> {
        Tensor<T, Rank> lhs{tensor.shape()};
        lhs.fill(lhs_value);
        return lhs + tensor;
    }

    template <typename T, std::size_t Rank>
    [[nodiscard]] constexpr auto operator-(const T& lhs_value, const Tensor<T, Rank>& tensor)
        -> Tensor<T, Rank> {
        Tensor<T, Rank> lhs{tensor.shape()};
        lhs.fill(lhs_value);
        return lhs - tensor;
    }

    template <typename T, std::size_t Rank>
    [[nodiscard]] constexpr auto operator*(const T& lhs_value, const Tensor<T, Rank>& tensor)
        -> Tensor<T, Rank> {
        Tensor<T, Rank> lhs{tensor.shape()};
        lhs.fill(lhs_value);
        return lhs * tensor;
    }

    template <typename T, std::size_t Rank>
    [[nodiscard]] constexpr auto operator/(const T& lhs_value, const Tensor<T, Rank>& tensor)
        -> Tensor<T, Rank> {
        Tensor<T, Rank> lhs{tensor.shape()};
        lhs.fill(lhs_value);
        return lhs / tensor;
    }

    template <typename T>
    constexpr auto operator<<(std::ostream& out, const Tensor<T, 1>& tensor) -> std::ostream& {
        std::ranges::copy(tensor, std::ostream_iterator<T>(out, " "));
        return out;
    }

    template <typename T>
    constexpr auto operator<<(std::ostream& out, const Tensor<T, 2>& tensor) -> std::ostream& {
        out << "{\n";

        for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
            for (std::size_t j = 0; j < tensor.shape()[1]; ++j) {
                out << tensor(i, j) << ' ';
            }
            out << '\n';
        }

        out << "}\n";
        return out;
    }

    template <typename T, std::size_t Rank>
    constexpr auto operator<<(std::ostream& out, const Tensor<T, Rank>& tensor) -> std::ostream& {
        out << "{\n";

        const std::size_t step = tensor.size() / tensor.shape()[0];

        for (std::size_t i = 0; i < tensor.shape()[0]; ++i) {
            std::array<std::size_t, Rank - 1> sub_shape;

            for (std::size_t j = 0; j < Rank - 1; ++j) {
                sub_shape[j] = tensor.shape()[j + 1];
            }

            const std::size_t start = i * step;

            Tensor<T, Rank - 1> sub_tensor(sub_shape);
            for (std::size_t j = 0; j < step; ++j) {
                sub_tensor[j] = tensor[start + j];
            }
            out << sub_tensor;
        }

        out << "}\n";
        return out;
    }
}  // namespace

#endif
