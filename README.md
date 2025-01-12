# Learning about LLMs

Playground project to learn about LLMs.

### Sources:

- [Andrej Karpathy's makemore project](https://www.youtube.com/watch?v=PaCmpygFfXo)

### Prerequisites:

- Python Jupyter Notebook
- Pytorch
- Matplotlib

## Table of Contents

- [Bigram LLM](./bigramllm.ipynb)
  - The main idea is to get a count of character pairs which occur in the text.
  - Arrange the dataset of the text so that the character pairs per letter are row-wise and column-wise (the second char in the pair in the col is the first char in the row)
  - Get a probability of a letter following another letter based on the character pairs in a row (Char Pair for a letter / Total Count of Char pair occurences for that letter)
  - repeat the loop since the column selected lines up with the starting char of the next pair by row index (repeat loop on that row)
- [Bigram LLM built with a Neural Network](./bigrams_neuralnetwork.ipynb)
- [Neural Networks](./NeuralNetworks/)
  - Following [Andrej Karpathy's building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
  - [Derivatives](./NeuralNetworks/derivatives.ipynb)
  - [Notebook](./NeuralNetworks/micrograd.ipynb)
    - Back Propagation
      - Chain Rule
