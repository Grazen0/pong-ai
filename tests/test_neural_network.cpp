#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "utec/algebra/tensor.h"
#include "utec/nn/activation.h"
#include "utec/nn/dense.h"
#include "utec/nn/loss.h"
#include "utec/nn/neural_network.h"

using utec::algebra::Tensor;

TEST_CASE("Dense forward/backward", "[dense]") {
    using utec::neural_network::Dense;
    using T = float;

    Dense<T> dense(2, 4);

    REQUIRE_THROWS_AS(dense.forward(Tensor<T, 2>(3, 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(dense.forward(Tensor<T, 2>(1, 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(dense.forward(Tensor<T, 2>(2, 0)), std::invalid_argument);
    REQUIRE_THROWS_AS(dense.forward(Tensor<T, 2>(2, 7)), std::invalid_argument);

    Tensor<T, 2> x(2, 1);
    x(0, 0) = 5;
    x(1, 0) = 2;

    dense.forward(x);
}

TEST_CASE("ReLU forward/backward", "[relu]") {
    using utec::neural_network::ReLU;
    using T = float;

    Tensor<T, 2> m(2, 2);
    m(0, 0) = -1;
    m(0, 1) = 2;
    m(1, 0) = 0;
    m(1, 1) = -3;

    ReLU<T> relu;
    Tensor<T, 2> r = relu.forward(m);

    REQUIRE(r(0, 0) == 0);
    REQUIRE(r(0, 1) == 2);
    REQUIRE(r(1, 0) == 0);
    REQUIRE(r(1, 1) == 0);

    Tensor<T, 2> gr(2, 2);
    gr.fill(1.0);

    Tensor<T, 2> dm = relu.backward(gr);

    REQUIRE(dm(0, 0) == 0);
    REQUIRE(dm(0, 1) == 1);
    REQUIRE(dm(1, 0) == 0);
    REQUIRE(dm(1, 1) == 0);
}

TEST_CASE("MSELoss forward/backward", "[loss]") {
    using utec::neural_network::MSELoss;
    using T = float;

    Tensor<T, 2> pred(1, 2);
    pred(0, 0) = 1;
    pred(0, 1) = 2;

    Tensor<T, 2> target(1, 2);
    target(0, 0) = 0;
    target(0, 1) = 4;

    MSELoss<T> loss;
    T l = loss.forward(pred, target);
    REQUIRE(l == 2.5);

    Tensor<T, 2> dp = loss.backward();
    REQUIRE(dp(0, 0) == 1);
    REQUIRE(dp(0, 1) == -2);
}

TEST_CASE("XOR training", "[neural_network]") {
    using utec::neural_network::Dense;
    using utec::neural_network::NeuralNetwork;
    using utec::neural_network::ReLU;
    using T = float;

    Tensor<T, 2> x(4, 2);
    x(0, 0) = 0, x(0, 1) = 0;
    x(1, 0) = 0, x(1, 1) = 1;
    x(2, 0) = 1, x(2, 1) = 0;
    x(3, 0) = 1, x(3, 1) = 1;

    Tensor<T, 2> y(4, 1);
    y(0, 0) = 0, y(1, 0) = 1;
    y(2, 0) = 1;
    y(3, 0) = 0;

    NeuralNetwork<T> net;
    net.add_layer(std::make_unique<Dense<T>>(2, 4));
    net.add_layer(std::make_unique<ReLU<T>>());
    net.add_layer(std::make_unique<Dense<T>>(4, 1));

    T final_loss = net.train(x, y, 1000, 0.1);
    REQUIRE(final_loss < 0.1);
}
