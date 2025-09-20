# Learning about LLMs

Playground project to learn about LLMs.

### Sources:

- [Andrej Karpathy's makemore project](https://www.youtube.com/watch?v=PaCmpygFfXo)

### Prerequisites:

- Python Jupyter Notebook
- Pytorch
- Matplotlib

## Table of Contents

### LLMs

- [LLM Guide and Overview](./LLM.md)
- [Bigram LLM](./bigramllm.ipynb)
  - The main idea is to get a count of character pairs which occur in the text.
  - Arrange the dataset of the text so that the character pairs per letter are row-wise and column-wise (the second char in the pair in the col is the first char in the row)
  - Get a probability of a letter following another letter based on the character pairs in a row (Char Pair for a letter / Total Count of Char pair occurences for that letter)
  - repeat the loop since the column selected lines up with the starting char of the next pair by row index (repeat loop on that row)
- [Bigram LLM built with a Neural Network](./NeuralNetworks/bigrams_neuralnetwork.ipynb)

### Neural Networks

- [Neural Networks](./NeuralNetworks/)
  - Following [Andrej Karpathy's building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
  - [Derivatives](./NeuralNetworks/derivatives.ipynb)
  - [Notebook](./NeuralNetworks/neural_network.ipynb)
    - Back Propagation using the Chain Rule
- [Regularization](./Regularization.md)
- [Makemore 2 (Andrej Karpathy)](./makemore2.md)

## Architecture of a Neural Network

- Made up of inputs, weights and bias that are inputs to layers of Neurons
- Loss is calculated after data passes through the layers
  - Mean squared error, Max-margin, Cross Entropy Loss, Negative Log Likelihood
  - For regression, use Mean squared error, for Classification use Negative Log Likelihood
- Back propagation pass is done to determine weight/bias adjustments needed to get closer to target output
- Gradient Descent: Loop back to running predictions with the upated weights and repeat Loss back propagation and parameter adjustments to continually lower the Loss

### Primary Components of a Neuron:

[Visual Model of a Neuron](https://cs231n.github.io/neural-networks-1/)

- $x_n$: Inputs to the neuron
- $w_n$: Weights (on the synapses)
- Processing in the Neuron: The set of weights multiplied by their corresponding inputs with a bias

  - what flows to the neuron are the multiple sets of inputs multiplied by the weights: $w_1 \times x_1, w_2 \times x_2, \ldots, w_n \times x_n$
  - Added to this is some bias $b$ which can be used to adjust the sensitivity or "trigger happiness" of the neuron regardless of the input.
    $$\sum_n w_n x_n + b$$
  - The product of the inputs, weights with the bias is piped to an Activation Function

    - The Activation Function is usually a [squashing function](https://en.wikipedia.org/wiki/Hyperbolic_functions) of some kind (Sigmoid, Relu or Tanh)
    - The squashing function squashes so that the output plateaus and caps smoothly at 1 or -1 (as the inputs are increased or decreased from zero):

      ![tanh function](./img/tanh.png)

  - The output of the neuron is the Activation function applied to the dot product of the weights/inputs+bias:
    $$f\left(\sum_n x_n w_n + b\right)$$

### Layer of Neurons

[Python Notebook](./NeuralNetworks/neural_network.ipynb)

- A set of Neurons evaluated independently

  - Each neuron in a layer is not connected to each other, but are connected to all other neurons or inputs in adjacent layers.

    ![Nueron Layers](./img/neural_net.jpeg)

### Multi-Layer Perceptron (MLP)

[Python Notebook](./NeuralNetworks/neural_network.ipynb)

- A network with multiple Layers of Neurons
- The Layers feed into each other sequentially (in order)

#### Note on Large Datasets

- In practice, for very large datasets, batching is done which takes a smaller subset of the data and uses that for the forward and backward pass
- See Andrew Karpathy's [Micrograd demo](https://github.com/karpathy/micrograd/blob/master/demo.ipynb) for example code

```Python
def loss(batch_size=None):

    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
```

## Papers

- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) - influential paper on Multi-Layer Perceptrons
  - Places words in multi dimensional vector embedding space where similar or related words are grouped closer together
  - Allows for predicting the next word if previous words were not present in the training set (you can find similar words in the vector space)

## Over-Fitting
 
- Can happen when you have very many parameters compared to too few examples in training. This makes the loss easy to make very low.
- The full dataset must be large enough to mitigate this.

## Batching

- In practice, to prevent iterations from taking a very long time, batches of the datasets are iterated over in Gradient Descent (see timestamp 42:24 at https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4) 
  - Select a random mini-set of data and forward and backward pass on that mini-batch
