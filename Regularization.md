# Regularization

- [Mike X Cohen's Deep Learning course](https://www.udemy.com/course/deeplearning_x/learn/lecture/27844198#content)

A method that penalizes the model from memorizing or over-learning from the training examples

- Helps the model generalize to new unseen samples
- Can change the nature and shape of the representations of the model
  - Can make it more sparse (relies on fewer units in the model) or more distributed
  - Changes the way the model represents the feature space

### Considerations

- Can increase or decrease the training time
- tends to decrease the training accuracy (except for state of the art models like image recognition), but helps increase generalization
- Works better with models that are larger and have many hidden layers
- Works better with sufficiently large enough dataset
  - not needed as much for small models with smaller datasets and not recommended to add regularization to smaller simpler models - can actually hurt

### Ways to Add Regularization

- Modify the model (dropout)
  - Dropout: remove nodes randomly during learning by forcing the activation/output to be 0
- Add a cost to the loss function (L1 or L2 regularization)
  - Adding something to the loss prevents the weights from getting too large
  - Makes sure the weights are staying in a reasonable range relatively close to 0
- Modify or add data (batch training, data augmentation)
  - Add more data, usually used for images/CNNs
  - Zoom, flip, change color etc. of the images to create more of them
- Figuring out which regularization method is best is found by trial and error and depends on specific model and data.
  - Often using any regularization method can yield comparable results, especially in more traditional machine learning outside of deep learning.

### Why Add Regularization?

- Adds a cost to the complexity of the solution which will drive the model to favor simpler solutions instead of memorizing the individual samples (more complex)
- smoothing for the solution (similar to above)
- Prevents model from learning item-specific details of the data samples
- \*Prevents over-fitting and makes the line through the data smoother instead of an exact fit to the training data

## Training vs. Evaluation

- When implementing some Regularization methods like Dropout, we need to switch the model between Training and Evaluation Mode

### Training Mode/Eval Mode

- Need to deactivate gradient computations and regularization when the model is being evaluated
  - During Training we use back propagation and the Gradients
  - `net.train()` - default
  - `net.eval()` - necessary when evaluating models with dropout or batch regularization only
  - `with torch.no_grad():` - not necessary, but adds overhead during testing which we should turn off for faster performance
- During Eval mode we do not use this
- Regularization methods like Dropout and Batch data are only applied to the model during training, not evaluation or testing of the model

## Dropout Regulatization

- set a probability $p$ for dropping out nodes (typically .5)
- At each training epoch (iteration), randombly pick 50% (or the rate of p) of the neurons/nodes and switch them off (they're output is forced to be 0 so they are not contributing to the model)
  - At epoch/iteration training 2, we pick a random different set of nodes/neurons to switch off
  - This continues for every training epoch
