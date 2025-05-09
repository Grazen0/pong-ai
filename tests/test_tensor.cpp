#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "../src/include/utec/algebra/Tensor.h"

using utec::algebra::Tensor;

TEST_CASE("creation, access and fill", "[tensor]") {
    Tensor<int, 2> tensor(2, 3);
    tensor.fill(7);
    int x = tensor(1, 2);
    REQUIRE(x == 7);
}

TEST_CASE("Valid reshape and linear access", "[tensor]") {
    Tensor<int, 2> tensor(2, 3);
    tensor.reshape(3, 2);
    int y = tensor[5];
    REQUIRE(y == tensor(2, 1));
}

TEST_CASE("3-dimensional indexing", "[tensor]") {
    Tensor<int, 3> tensor(3, 5, 2);

    tensor[7] = 42;
    tensor[12] = 38;
    tensor[20] = 256;
    tensor[21] = 100;

    REQUIRE(tensor(0, 3, 1) == 42);
    REQUIRE(tensor(1, 1, 0) == 38);
    REQUIRE(tensor(2, 0, 0) == 256);
    REQUIRE(tensor(2, 0, 1) == 100);
}

TEST_CASE("3-dimensional assigning", "[tensor]") {
    Tensor<int, 3> tensor(3, 5, 2);

    tensor(0, 3, 1) = 42;
    tensor(1, 1, 0) = 38;
    tensor(2, 0, 0) = 256;
    tensor(2, 0, 1) = 100;

    REQUIRE(tensor[7] == 42);
    REQUIRE(tensor[12] == 38);
    REQUIRE(tensor[20] == 256);
    REQUIRE(tensor[21] == 100);
}

TEST_CASE("Invalid reshape", "[tensor]") {
    Tensor<int, 3> t3(2, 2, 2);
    REQUIRE_THROWS_AS(t3.reshape(2, 4, 2), std::invalid_argument);
}

TEST_CASE("Tensor addition and subtraction", "[tensor]") {
    Tensor<double, 2> tensor_a(2, 2);
    Tensor<double, 2> tensor_b(2, 2);

    tensor_a(0, 1) = 5.5;
    tensor_b.fill(2.0);

    auto sum = tensor_a + tensor_b;
    auto diff = sum - tensor_b;

    REQUIRE(sum(0, 1) == 7.5);
    REQUIRE(diff(0, 1) == 5.5);
}

// TEST_CASE("2d transpose", "[tensor]") {
//     Tensor<int, 2> mat(3, 2);
//     Tensor<int, 2> mat_t = mat.transpose_2d();
// }
