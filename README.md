# Cat vs Dog Image Classification using Logistic Regression
## !!! It is suitable for running on the Kaggle platform. !!!

This project demonstrates a binary image classification task where the goal is to classify images of cats and dogs using Logistic Regression. The dataset contains labeled images of cats and dogs and is processed for training a logistic regression model to distinguish between the two categories.

## Dataset

The dataset used in this project is the "Cat vs Dog" dataset, available on Kaggle. It is split into two directories:

- **train/**: Contains training images of cats and dogs.
- **test/**: Contains test images of cats and dogs.

The data is organized as follows:

- **train/cat/**: Training images of cats
- **train/dog/**: Training images of dogs
- **test/cat/**: Test images of cats
- **test/dog/**: Test images of dogs

## Project Structure

- **Libraries**: 
  - `numpy` for numerical operations
  - `pandas` for data handling
  - `matplotlib` for data visualization
  - `seaborn` for advanced plotting
  - `PIL` for image processing
  - `cv2` for image manipulation
  - `os` for file handling
  - `sklearn` for train/test split

- **Functions**:
  - `initialize_weights_and_bias`: Initializes weights and bias for the logistic regression model.
  - `sigmoid`: Applies the sigmoid activation function.
  - `forward_backward_propagation`: Performs forward and backward propagation for cost calculation and gradient updates.
  - `update`: Updates the weights and bias using gradient descent.
  - `predict`: Makes predictions based on the trained model.
  - `logistic_regression`: Main function to train the model and evaluate its accuracy.

## Installation

To run this project, you will need to install the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python scikit-learn
```

## How to Run

1. Download the dataset and place it in the appropriate directories (`train/cat`, `train/dog`, `test/cat`, `test/dog`).
2. Run the script to preprocess the images, train the logistic regression model, and evaluate its accuracy.

```bash
python cat_dog_classification.py
```

## Preprocessing

- The images are resized to 50x50 pixels and converted to grayscale.
- Both training and test images are processed to create training and test datasets, with labels 1 for cats and 0 for dogs.

## Model

The logistic regression model is used for binary classification. The following steps are performed:

1. **Initialization**: Weights and bias are initialized.
2. **Training**: The model is trained using gradient descent to minimize the cost function.
3. **Prediction**: The model predicts whether an image is of a cat or a dog.
4. **Evaluation**: The model's accuracy is calculated on both the training and test datasets.

## Results

The accuracy of the model is printed after training. Both training and test accuracies are displayed as percentages.

## Conclusion

This project demonstrates the application of logistic regression to image classification. Despite its simplicity, the logistic regression model can perform reasonably well on this dataset, especially with a well-preprocessed input.
