# Exercise 4.1: Ubah program Example 4.1 supaya melakukan logika OR

from numpy import exp, array, random, dot

# Kolom ketiga = bias (selalu 1)
inputs = array([[0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])
outputs = array([[0, 1, 1, 1]]).T          # tabel kebenaran OR
random.seed(1)
weights = 2 * random.random((3, 1)) - 1


def neuron(inputs, weights):
    return 1 / (1 + exp(-(dot(inputs, weights))))


for iteration in range(50000):
    output = neuron(inputs, weights)
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test semua kombinasi input
print("A B | A OR B (output neuron)")
for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    x = array([a, b, 1])
    print(a, b, "|", round(float(neuron(x, weights)[0]), 4))
