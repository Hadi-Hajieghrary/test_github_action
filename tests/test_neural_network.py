import torch
from my_package.neural_network import NeuralNetwork


def test_neural_network():
    model = NeuralNetwork()
    sample_input = torch.randn(1, 1, 28, 28)
    output = model(sample_input)
    assert output.shape == torch.Size([1, 10])


if __name__ == "__main__":
    test_neural_network()