# Fashion Classification Neural Network

This GitHub repository contains the implementation of a neural network for solving a classification problem on the Fashion MNIST dataset. The project is divided into three main problems, each addressing different aspects of the neural network implementation, hyperparameter tuning, and optimization visualization.

## 1. Neural Network Implementation

In this section, we implement a neural network from scratch using NumPy. Key components include the definition of the network architecture, hyperparameters, weight initialization, activation functions (ReLU and softmax), and the cross-entropy loss function. The network is trained using stochastic gradient descent (SGD).

## 2. Hyperparameter Tuning with TensorFlow

The second problem focuses on optimizing the neural network's hyperparameters using TensorFlow. We define a flexible neural network model in TensorFlow, conduct a search over various hyperparameter combinations, and identify the set that results in the best validation accuracy. The final model is trained with the optimal hyperparameters and evaluated on the test set. Training loss over the last 20 epochs is visualized.

## 3. Visualization of SGD Trajectory

The third problem involves visualizing the stochastic gradient descent (SGD) optimization trajectory using principal component analysis (PCA). We apply PCA to the trajectory of SGD optimization and create a 3D plot to illustrate the loss landscape. The SGD trajectory is overlaid on the plot, providing insights into the optimization process.

## Project Structure

- `fashion-classification.ipynb`: Jupyter Notebook containing the entire project code.
- `fashion_mnist_train_images.npy`: [Training images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy) for the Fashion MNIST dataset.
- `fashion_mnist_train_labels.npy`: [Training labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy) for the Fashion MNIST dataset.
- `fashion_mnist_test_images.npy`: [Test images](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy) for the Fashion MNIST dataset.
- `fashion_mnist_test_labels.npy`: [Test labels](https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy) for the Fashion MNIST dataset.

## Usage

To run the code and explore the implementation, open the `fashion-classification.ipynb` notebook in a Jupyter environment. Ensure that the required dependencies, such as NumPy, Matplotlib, scipy, scikit-learn, and TensorFlow, are installed in your Python environment.
