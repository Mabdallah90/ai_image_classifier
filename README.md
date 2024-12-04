# AI Image Classifier

This project is a convolutional neural network (CNN)-based image classification system implemented in Python using PyTorch. The system classifies images into one of several predefined categories, with support for grayscale conversion, data augmentation, and model checkpointing.

## Features

- **CNN Architecture**: A custom deep learning model (`MyCNN`) with multiple convolutional, batch normalization, and pooling layers.
- **Dataset Handling**: A custom dataset class (`ImagesDataset`) for loading and preprocessing images, including grayscale conversion and resizing.
- **Training and Validation**: Script for model training, validation, and checkpointing to save the best-performing model.
- **Metrics**: Displays training and validation accuracy and loss after each epoch.
- **Adaptable**: Designed to handle grayscale images with configurable input dimensions.

## Files

- **`architecture.py`**: Defines the CNN model architecture.
- **`dataset.py`**: Contains the dataset class for handling images and their labels.
- **`main.py`**: Handles model training, validation, and dataset loading.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- PIL (Pillow)
- scikit-learn

You can install the dependencies using:

```bash
pip install torch numpy pillow scikit-learn
```

## How to Use

1. **Prepare the Dataset**:
   - Organize your images in a directory along with a `.csv` file specifying their labels. The CSV file should follow this format:
     ```
     filename;label
     image1.jpg;class1
     image2.jpg;class2
     ...
     ```
   - Ensure the images are in `.jpg` format and the CSV file is in the same directory.

2. **Set Up the Project**:
   - Clone or download this repository:
     ```bash
     git clone https://github.com/your-repo/image-classifier.git
     cd image-classifier
     ```
   - Place your dataset in a directory (e.g., `./training_data`) or update the dataset path in `main.py`:
     ```python
     main_data = ImagesDataset('./training_data')
     ```

3. **Train the Model**:
   - Run the training script:
     ```bash
     python main.py
     ```
   - The script will train the model, validate its performance, and save the best model as `model.pth` in the current directory.

4. **Evaluate the Model**:
   - To test the model on new data:
     - Load the saved model:
       ```python
       model.load_state_dict(torch.load("model.pth"))
       model.eval()
       ```
     - Use the `ImagesDataset` class to load test data and make predictions.

5. **Customize Hyperparameters**:
   - You can modify the number of epochs, learning rate, or batch size in `main.py`:
     ```python
     num_epochs = 35
     optimizer = optim.Adam(model.parameters(), lr=0.001)
     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
     ```

6. **Adjust Model Settings**:
   - If your dataset contains a different number of classes, update the last fully connected layer in `architecture.py`:
     ```python
     self.fc2 = nn.Linear(512, <number_of_classes>)
     ```

