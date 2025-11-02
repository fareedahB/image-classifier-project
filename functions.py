import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from torchvision import datasets, transforms, models
from collections import Counter
from PIL import Image
import numpy as np
import json

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir):
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  # Define your transforms for the training, validation, and testing sets
  train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
  valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
  test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

  # Load the datasets with ImageFolder
  train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
  valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
  test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

  # Using the image datasets and the trainforms, define the dataloaders
  class_counts = Counter(train_data.targets)  # Count occurrences of each class
  num_classes = len(class_counts)
  # Compute class weights (inverse of frequency)
  class_weights = torch.tensor(
      [1.0 / class_counts[c] for c in range(num_classes)], dtype=torch.float32).to(device)
  # Compute sample weights
  sample_weights = [1.0 / class_counts[c] for c in train_data.targets]
  # Create a sampler
  sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

  # Use the sampler in the DataLoader
  trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=sampler)
  validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
  testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

  return trainloader, validloader, testloader

def build_model(architecture, hidden_layer_1_units, hidden_layer_2_units):
  # Build your network
  # Load a pre-trained ResNet50 model
  model = models.resnet50(pretrained=True)

  # Freeze the parameters of the pre-trained model to prevent them from being updated during training
  for param in model.parameters():
      param.requires_grad = False

  # Define a new classifier to replace the last fully connected layer of the model
  classifier = nn.Sequential(
    nn.Linear(2048, hidden_layer_1_units),
    nn.ReLU(),
    nn.Linear(hidden_layer_1_units, hidden_layer_2_units),
    nn.ReLU(),
    nn.Linear(hidden_layer_2_units, 102),
    nn.LogSoftmax(dim=1)
  )

  # Replace the fully connected layer of the model with the new classifier
  model.fc = classifier

  return model


def train_model(model, trainloader, validloader, epochs, learning_rate, gpu):
  # Set the number of epochs for training
  step = 0
  running_loss = 0
  print_every = 50
  criterion = nn.NLLLoss()
  optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate)

  # Move the model to the specified device (GPU or CPU)
  model.to(device)# Print training progress every 50 steps
  # Training loop
  for epoch in range(epochs):
      for images, labels in trainloader:
          step += 1

          # Move images and labels to the specified device
          images, labels = images.to(device), labels.to(device)

          # Zero the gradients for the optimizer
          optimizer.zero_grad()

          # Forward pass: compute the model output
          logps = model(images)

          # Calculate the loss using the predicted outputs and true labels
          loss = criterion(logps, labels)

          # Backward pass: compute gradients
          loss.backward()

          # Update the model parameters
          optimizer.step()

          # Accumulate the running loss
          running_loss += loss.item()

          # Evaluate the model every 'print_every' steps
          if step % print_every == 0:
              model.eval()  # Set the model to evaluation mode
              valid_loss = 0
              accuracy = 0

              # Loop through the validation data
              for images, labels in validloader:
                  # Move validation images and labels to the specified device
                  images, labels = images.to(device), labels.to(device)

                  # Forward pass: compute the model output for validation data
                  logps = model(images)

                  # Calculate the loss for validation data
                  loss = criterion(logps, labels)
                  valid_loss += loss.item()

                  # Calculate probabilities from log probabilities
                  ps = torch.exp(logps)

                  # Get the top predicted class
                  top_ps, top_class = ps.topk(1, dim=1)

                  # Check if the predicted class matches the true labels
                  equality = top_class == labels.view(*top_class.shape)

                  # Calculate accuracy
                  accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

              # Print training and Validation statistics
              print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")

              # Reset the running loss for the next print interval
              running_loss = 0

              # Set the model back to training mode
              model.train()
  return optimizer
# Save the checkpoint
# Attach class_to_idx to the model
def save_checkpoint(model, optimizer, epochs, class_to_idx):
    # Define the checkpoint dictionary
    checkpoint = {
      'model_state_dict': model.state_dict(),  # Model parameters
      'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
      'epochs': epochs,  # Number of epochs trained
      'class_to_idx': class_to_idx  # Save class mapping
      }

    # Save the checkpoint
    torch.save(checkpoint, 'checkpoint.pth')

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    # Rebuild the model architecture
    model = models.resnet50(pretrained=True)

    # Replace the classifier with the one from the checkpoint
    classifier = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 102),
        nn.LogSoftmax(dim=1)
    )
    model.fc = classifier

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']

    # Load optimizer state
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    # Open the image
    image = Image.open(image)

    image.resize((256, 256)) # Resize

    # Center Crop to 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert to NumPy array & normalize
    np_image = np.array(image) / 255.0

    # Normalize using mean and standard deviation per channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions (H, W, C) → (C, H, W)
    np_image = np_image.transpose((2, 0, 1))

    tensor_image = torch.tensor(np_image, dtype=torch.float32)

    return tensor_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    # Process the image
    image = process_image(image_path)

    # Convert to a PyTorch tensor and add batch dimension
    image = image.unsqueeze(0)  # Shape: (1, C, H, W)

    # Ensure model is in evaluation mode & move image to same device as model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)

    # Forward pass (disable gradient computation for efficiency)
    with torch.no_grad():
        output = model(image)

    # Convert log probabilities to actual probabilities using softmax
    probabilities = torch.exp(output)

    # Get the top-K probabilities and their corresponding class indices
    top_probs, top_indices = probabilities.topk(topk, dim=1)

    # Convert tensors to lists
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    # Invert class_to_idx dictionary to map indices to actual class labels
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes
