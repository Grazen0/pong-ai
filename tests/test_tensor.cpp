#include <utec/algebra/Tensor.h>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

TEST_CASE("Creation, access and fill", "[tensor]") {
    utec::algebra::Tensor<int, 2> t(2L, 3L);
    t.fill(7);
    int x = t(1, 2);
    REQUIRE(x == 7);
}

TEST_CASE("Valid reshape and linear access", "[tensor]") {
    utec::algebra::Tensor<int, 2> t2{2, 3};
    t2.reshape(3, 2);
    int y = t2[5];
    REQUIRE(y == t2(2, 1));
}

TEST_CASE("Invalid reshape", "[tensor]") {
    utec::algebra::Tensor<int, 3> t3(2, 2, 2);
    REQUIRE_THROWS_AS(t3.reshape(2, 4, 2), std::invalid_argument);
}

TEST_CASE("Tensor addition and subtraction", "[tensor]") {
    utec::algebra::Tensor<double, 2> a(2, 2), b(2, 2);
    a(0, 1) = 5.5;
    b.fill(2.0);
    auto sum = a + b;
    auto diff = sum - b;
    REQUIRE(sum(0, 1) == 7.5);
    REQUIRE(diff(0, 1) == 5.5);
}
