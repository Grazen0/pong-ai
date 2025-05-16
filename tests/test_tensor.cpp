#include <array>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "../src/include/utec/algebra/tensor.h"

using utec::algebra::Tensor;

TEST_CASE("2d tensor basic operations", "[tensor]") {
    Tensor<int, 2> tensor(2, 3);
    tensor[0] = 1;
    tensor[1] = 2;
    tensor[2] = 3;
    tensor[3] = 4;
    tensor[4] = 5;
    tensor[5] = 6;

    SECTION("fill", "[tensor]") {
        tensor.fill(7);

        REQUIRE(tensor[0] == 7);
        REQUIRE(tensor[1] == 7);
        REQUIRE(tensor[2] == 7);
        REQUIRE(tensor[3] == 7);
        REQUIRE(tensor[4] == 7);
        REQUIRE(tensor[5] == 7);
    }

    SECTION("valid indexing", "[tensor]") {
        REQUIRE(tensor(0, 0) == 1);
        REQUIRE(tensor(0, 1) == 2);
        REQUIRE(tensor(0, 2) == 3);
        REQUIRE(tensor(1, 0) == 4);
        REQUIRE(tensor(1, 1) == 5);
        REQUIRE(tensor(1, 2) == 6);
    }

    SECTION("invalid indexing", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, 42), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(-1, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, 3), std::out_of_range);
    }

    SECTION("valid reshape", "[tensor]") {
        tensor.reshape(3, 2);

        REQUIRE(tensor.shape() == std::array<size_t, 2>{3, 2});
        REQUIRE(tensor(0, 0) == 1);
        REQUIRE(tensor(0, 1) == 2);
        REQUIRE(tensor(1, 0) == 3);
        REQUIRE(tensor(1, 1) == 4);
        REQUIRE(tensor(2, 0) == 5);
        REQUIRE(tensor(2, 1) == 6);
    }

    SECTION("invalid reshape", "[tensor]") {
        REQUIRE_THROWS_AS(tensor.reshape(2, 4), std::invalid_argument);
        REQUIRE_THROWS_AS(tensor.reshape(3, 3), std::invalid_argument);
        REQUIRE_THROWS_AS(tensor.reshape(1, 1), std::invalid_argument);
        REQUIRE_THROWS_AS(tensor.reshape(3, 1), std::invalid_argument);
    }

    SECTION("scalar multiplication", "[tensor]") {
        Tensor<int, 2> prod = tensor * 3;

        REQUIRE(prod(0, 0) == 3);
        REQUIRE(prod(0, 1) == 6);
        REQUIRE(prod(0, 2) == 9);
        REQUIRE(prod(1, 0) == 12);
        REQUIRE(prod(1, 1) == 15);
        REQUIRE(prod(1, 2) == 18);
    }

    SECTION("transpose", "[tensor]") {
        Tensor<int, 2> mat(3, 2);
        mat(0, 0) = 1;
        mat(0, 1) = 2;
        mat(1, 0) = 3;
        mat(1, 1) = 4;
        mat(2, 0) = 5;
        mat(2, 1) = 6;

        Tensor<int, 2> mat_t = mat.transpose_2d();

        REQUIRE(mat_t.shape() == std::array<size_t, 2>{2, 3});
        REQUIRE(mat_t(0, 0) == 1);
        REQUIRE(mat_t(0, 1) == 3);
        REQUIRE(mat_t(0, 2) == 5);
        REQUIRE(mat_t(1, 0) == 2);
        REQUIRE(mat_t(1, 1) == 4);
        REQUIRE(mat_t(1, 2) == 6);
    }
}

TEST_CASE("2d tensor binary operations", "[tensor]") {
    Tensor<int, 2> tensor_a(2, 3);
    tensor_a[0] = 7;
    tensor_a[1] = 17;
    tensor_a[2] = 31;
    tensor_a[3] = 0;
    tensor_a[4] = 63;
    tensor_a[5] = 102;

    Tensor<int, 2> tensor_b(2, 3);
    tensor_b[0] = 3;
    tensor_b[1] = 5;
    tensor_b[2] = -2;
    tensor_b[3] = 5;
    tensor_b[4] = 6;
    tensor_b[5] = 42;

    SECTION("addition", "[tensor]") {
        Tensor<int, 2> sum = tensor_a + tensor_b;

        REQUIRE(sum(0, 0) == 10);
        REQUIRE(sum(0, 1) == 22);
        REQUIRE(sum(0, 2) == 29);
        REQUIRE(sum(1, 0) == 5);
        REQUIRE(sum(1, 1) == 69);
        REQUIRE(sum(1, 2) == 144);
    }

    SECTION("subtraction", "[tensor]") {
        Tensor<int, 2> diff = tensor_a - tensor_b;

        REQUIRE(diff(0, 0) == 4);
        REQUIRE(diff(0, 1) == 12);
        REQUIRE(diff(0, 2) == 33);
        REQUIRE(diff(1, 0) == -5);
        REQUIRE(diff(1, 1) == 57);
        REQUIRE(diff(1, 2) == 60);
    }
}

TEST_CASE("3d tensor", "[tensor]") {
    Tensor<int, 3> tensor(3, 5, 2);
    tensor[7] = 42;
    tensor[12] = 38;
    tensor[20] = 256;
    tensor[21] = 100;

    SECTION("valid indexing", "[tensor]") {
        REQUIRE(tensor(0, 3, 1) == 42);
        REQUIRE(tensor(1, 1, 0) == 38);
        REQUIRE(tensor(2, 0, 0) == 256);
        REQUIRE(tensor(2, 0, 1) == 100);
    }

    SECTION("invalid indexing", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(4, 0, 0), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 2, -1), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 5, 29), std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, -4, 2), std::out_of_range);
    }

    SECTION("valid assigning", "[tensor]") {
        tensor(0, 3, 1) = 1;
        tensor(1, 1, 0) = 2;
        tensor(2, 0, 0) = 3;
        tensor(2, 0, 1) = 4;

        REQUIRE(tensor[7] == 1);
        REQUIRE(tensor[12] == 2);
        REQUIRE(tensor[20] == 3);
        REQUIRE(tensor[21] == 4);
    }

    SECTION("invalid assigning", "[tensor]") {
        REQUIRE_THROWS_AS(tensor(4, 0, 0) = 1, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 2, -1) = 2, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(1, 5, 29) = 3, std::out_of_range);
        REQUIRE_THROWS_AS(tensor(2, -4, 2) = 4, std::out_of_range);
    }
}
