# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
<img width="1092" height="787" alt="image" src="https://github.com/user-attachments/assets/555bd731-5a89-4abe-bd0b-4fe0282b500a" />


## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.
### STEP 3: 
Visualize sample images from the dataset.
### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.
### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.
### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions. 




## PROGRAM

### Name: Parthipan M

### Register Number: 212225040294

```python
# Step 1: Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import zipfile
import os

# Step 2: Upload and Extract Dataset
from google.colab import files

uploaded = files.upload()
zip_file = next(iter(uploaded))

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("./data")

print("Dataset extracted successfully!")

# Step 3: Load and Preprocess Data
dataset_path = "./data/dataset/"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)

# Step 4: Display Sample Images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))

    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")

    plt.show()

show_sample_images(train_dataset)

# Step 5: Dataset Information
print("Total number of training samples:", len(train_dataset))

first_image, label = train_dataset[0]
print("Shape of the first image:", first_image.shape)

# Step 6: Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Step 7: Load Pretrained VGG19 Model
model = models.vgg19(
    weights=models.VGG19_Weights.DEFAULT
)

print(model)

# Step 8: Move Model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 9: Model Summary
!pip install torchsummary

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Step 10: Modify Final Layer
num_classes = len(train_dataset.classes)

in_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(
    in_features, num_classes
)

print("Number of classes:", num_classes)
print("Class names:", train_dataset.classes)

# Step 11: Move Model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 12: Freeze Layers and Optimizer
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# Step 13: Train the Model
def train_model(model, train_loader, test_loader, num_epochs=10):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

        val_loss = val_running_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}')

    # Plot training and validation loss
    print("Name: ARAVINDAN SD")
    print("Register Number: 212224243001")

    plt.figure(figsize=(8, 6))

    plt.plot(range(1, num_epochs + 1), train_losses,
             label='Train Loss', marker='o')

    plt.plot(range(1, num_epochs + 1), val_losses,
             label='Validation Loss', marker='s')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return train_losses, val_losses


# Step 14: Train the Model
train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
)

# Step 15: Test Model and Confusion Matrix
def test_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("Name: ARAVINDAN SD")
    print("Register Number: 212224243001")

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Name:        ")
    print("Register Number:        ")

    print("Classification Report:")

    print(classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.classes
    ))


# Step 16: Evaluate Model
test_model(model, test_loader)

# Step 17: Predict Single Image
def predict_image(model, image_index, dataset):

    model.eval()

    image, label = dataset[image_index]

    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        predicted = torch.argmax(outputs, dim=1).item()

    class_names = dataset.classes

    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)

    plt.title(
        f'Actual: {class_names[label]}\n'
        f'Predicted: {class_names[predicted]}'
    )

    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')


# Step 18: Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)
```

### OUTPUT

## Training Loss per Epoch

Include the Training Loss per epoch

## Confusion Matrix

Include confusion matrix here

## Classification Report
Include classification report here

### New Sample Data Prediction
Include your sample input and output here

## RESULT
Include your result here
