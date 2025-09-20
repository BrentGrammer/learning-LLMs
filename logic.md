Let's break down the logic behind how the probabilities are calculated and accessed in this neural network implementation for bigram language modeling:

Understanding the Probability Matrix (probs)
One-hot Encoding (xenc):
xenc is created by converting each input character (xs) into a one-hot vector. This means for each character in xs, you get a vector where all elements are zero except for one position which is 1, corresponding to the character's index in stoi.
Logits and Scores:
logits = xenc @ W: Here, each one-hot vector is multiplied by the weight matrix W. This operation gives you raw, unnormalized scores (logits) for each possible next character given the current character. The shape of logits will be [number of examples, NUM_CHARS].
Softmax Transformation:
scores = logits.exp(): Exponentiation to get positive scores.
probs = scores / scores.sum(1, keepdims=True): Softmax is applied to turn these scores into probabilities that sum to 1 for each input example. Now, probs is a matrix where probs[i, j] represents the probability of the j-th character following the i-th input character.

Indexing into probs
Row Index: Each row in probs corresponds to one example from xs. If xs[i] represents the character at index i, then row i in probs gives the probabilities of what character could follow this character.
Column Index: Each column in probs corresponds to a possible next character from the vocabulary (NUM_CHARS). 
Accessing Probability for the Actual Next Character:
probs[torch.arange(xs.nelement()), ys] does the following:
torch.arange(xs.nelement()) creates an array [0, 1, 2, ..., n-1] where n is the number of examples. This selects the correct row for each example.
ys contains the indices of the characters that actually followed each xs in the training data. Therefore, ys[i] is the correct column for the i-th example.
This indexing operation thus picks out the probability that the model assigned to the true next character for each input character.

Why This Works:
The structure of probs is designed such that for any given input character (represented by a row), the probabilities across the columns tell you the likelihood of each possible next character. 
By selecting probs[i, ys[i]] for each i, you're essentially checking how well the model predicted the actual next character (ys[i]) given the current character (xs[i]).

This approach is fundamental in training language models where the goal is to maximize the probability of the next correct character, thereby reducing the cross-entropy loss. Here, the indexing directly aligns with this goal by focusing on the exact probabilities we're interested in for loss calculation.