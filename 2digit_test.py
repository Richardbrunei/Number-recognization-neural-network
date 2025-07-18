import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import glob
import random
import numpy as np

# --- 1. New Network Architecture (Convolutional Neural Network) ---
class ConvNet(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed for two-digit recognition.
    It takes a 28x56 grayscale image as input (two 28x28 digits concatenated).
    The network has two output heads, one for each digit, allowing it to
    predict both digits simultaneously.
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers
        # Input: 1 channel (grayscale), Output: 32 channels, Kernel size: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer (reduces spatial dimensions)
        self.pool = nn.MaxPool2d(2, 2) # 2x2 window, stride 2

        # Fully connected layers
        # After two pooling layers, the 28x56 image becomes 7x14 features
        # 64 channels * 7 height * 14 width = 6272 features
        self.fc1 = nn.Linear(64 * 7 * 14, 128)
        # Two separate output heads for each digit
        self.fc_digit1 = nn.Linear(128, 10) # 10 classes for digit 0-9
        self.fc_digit2 = nn.Linear(128, 10) # 10 classes for digit 0-9

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x))) # Output size: 14x28
        x = self.pool(F.relu(self.conv2(x))) # Output size: 7x14

        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 64 * 7 * 14) # -1 infers batch size

        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))

        # Pass through the two separate output heads
        output_digit1 = F.log_softmax(self.fc_digit1(x), dim=1)
        output_digit2 = F.log_softmax(self.fc_digit2(x), dim=1)

        return output_digit1, output_digit2

# --- 2. Synthetic Two-Digit Dataset ---
class TwoDigitMNIST(Dataset):
    """
    A custom Dataset class to generate synthetic two-digit images
    by concatenating two MNIST digits horizontally.
    """
    def __init__(self, mnist_dataset, transform=None):
        self.mnist_dataset = mnist_dataset
        self.transform = transform

    def __len__(self):
        # We can generate a large number of combinations.
        # For simplicity, let's say we can generate as many as the square
        # of the original MNIST dataset size, but we'll limit it for practical
        # purposes to avoid excessive memory usage or training time.
        # Let's cap it at 60,000 for train and 10,000 for test, similar to MNIST size.
        return len(self.mnist_dataset) # Each call will generate a new pair

    def __getitem__(self, idx):
        # Get two random indices to pick two digits from the MNIST dataset
        idx1 = random.randint(0, len(self.mnist_dataset) - 1)
        idx2 = random.randint(0, len(self.mnist_dataset) - 1)

        img1, label1 = self.mnist_dataset[idx1]
        img2, label2 = self.mnist_dataset[idx2]

        # Convert PIL images to tensors if transform is not None, then concatenate
        # If transform is None, assume img1 and img2 are already PIL images
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            # Ensure images are 2D tensors (e.g., 1x28x28) before concatenating
            # If they are already 1x28x28, squeeze will remove the channel dim
            # then unsqueeze adds it back after concatenation.
            # Or, if they are PIL, convert to numpy arrays first
            if isinstance(img1, Image.Image):
                img1 = np.array(img1)
            if isinstance(img2, Image.Image):
                img2 = np.array(img2)
        else:
            # Assume images are already tensors if no transform is provided
            # and they are coming from a DataLoader that applies ToTensor
            pass

        # Concatenate images horizontally
        # Ensure images are 2D (height, width) before hstack for numpy
        # If they are 1x28x28, squeeze to 28x28 first
        if isinstance(img1, torch.Tensor):
            # If already tensor, assume channel first (C, H, W)
            # Cat along the width dimension (dim=2)
            combined_img = torch.cat((img1, img2), dim=2)
        else: # Assume numpy array (H, W) or PIL image converted to numpy
            combined_img = np.hstack((img1, img2))
            combined_img = Image.fromarray(combined_img) # Convert back to PIL for transform

        # Apply the transform to the combined image if provided
        if self.transform:
            combined_img = self.transform(combined_img)
            # Ensure it's a single channel if it somehow became 3 channels
            if combined_img.shape[0] == 3:
                combined_img = combined_img.mean(dim=0, keepdim=True)


        # Labels are returned as a tensor of two integers
        labels = torch.tensor([label1, label2], dtype=torch.long)

        return combined_img, labels

def get_two_digit_MNIST_loaders(is_train, batch_size=15, shuffle=True, invert_colors=False):
    """
    Helper function to get DataLoader for the two-digit MNIST dataset.
    """
    transform_list = [transforms.ToTensor()]
    if invert_colors:
        transform_list.append(transforms.Lambda(lambda x: ImageOps.invert(x)))
        transform_list.append(transforms.ToTensor())

    # Ensure the transform is applied to individual MNIST images before combining
    # and then to the combined image.
    # For simplicity, we'll apply ToTensor here and then handle concatenation
    # within TwoDigitMNIST to ensure it's done on tensor data.
    mnist_base_transform = transforms.Compose([transforms.ToTensor()])

    # Load the base MNIST dataset
    mnist_base_dataset = MNIST("", is_train, transform=mnist_base_transform, download=True)

    # Create the two-digit dataset
    # The transform here will be applied to the *combined* image
    two_digit_transform = transforms.Compose([
        transforms.ToPILImage(), # Convert tensor back to PIL for image ops if needed
        transforms.Grayscale(num_output_channels=1), # Ensure single channel
        transforms.ToTensor() # Convert back to tensor
    ])

    two_digit_dataset = TwoDigitMNIST(mnist_base_dataset, transform=two_digit_transform)

    return DataLoader(two_digit_dataset, batch_size=batch_size, shuffle=shuffle)

# --- 3. Evaluation Function ---
def evaluate(test_data, net):
    """
    Evaluates the network's accuracy on two-digit numbers.
    A prediction is considered correct only if *both* digits are predicted correctly.
    """
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            # x is the combined image, y is a tensor like [label1, label2]
            output_digit1, output_digit2 = net.forward(x)

            # Get predicted digits
            predicted_digit1 = torch.argmax(output_digit1, dim=1)
            predicted_digit2 = torch.argmax(output_digit2, dim=1)

            # Compare with true labels (y[:, 0] for first digit, y[:, 1] for second)
            for i in range(len(y)):
                if predicted_digit1[i] == y[i, 0] and predicted_digit2[i] == y[i, 1]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

# --- Main Program Logic ---
def main():
    print("Program start: Two-Digit MNIST Recognizer")

    # Define a path for saving the model
    model_save_path = 'two_digit_mnist_convnet.pth'

    # Initialize the new ConvNet
    net = ConvNet()

    # Check if a saved model exists
    loaded = False
    try:
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained model from {model_save_path}...")
            net.load_state_dict(torch.load(model_save_path))
            print("Model loaded successfully.")
            loaded = True
        else:
            print("No pre-trained model found. Training a new model...")
    except Exception as e:
        print(f"Unsuccessful loading: {e}. Creating new model...")
        loaded = False

    # Load test data for initial evaluation
    test_data = get_two_digit_MNIST_loaders(is_train=False)
    print("Initial accuracy:", evaluate(test_data, net))

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if loaded:
        inpt = input("More training? (y/n) ").strip()
        if inpt.lower() == 'y' or inpt.lower() == 'yes':
            loaded = False

    if not loaded:
        train_data = get_two_digit_MNIST_loaders(is_train=True)
        train_data_inv = get_two_digit_MNIST_loaders(is_train=True, invert_colors=True)

        try:
            num_epochs = int(input("How many epochs for training? ").strip())
        except ValueError:
            print("Invalid input. Defaulting to 5 epochs.")
            num_epochs = 5

        for epoch in range(num_epochs):
            for (x, y) in train_data:
                net.zero_grad()
                output_digit1, output_digit2 = net.forward(x)
                # Calculate loss for each digit and sum them
                loss1 = F.nll_loss(output_digit1, y[:, 0])
                loss2 = F.nll_loss(output_digit2, y[:, 1])
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            # Optional: Train with inverted colors (can help with robustness)
            for (x, y) in train_data_inv:
                net.zero_grad()
                output_digit1, output_digit2 = net.forward(x)
                loss1 = F.nll_loss(output_digit1, y[:, 0])
                loss2 = F.nll_loss(output_digit2, y[:, 1])
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} accuracy: {evaluate(test_data, net):.4f}")

        # Save the trained model
        inpt = input("Save trained model? (y/n) ").strip()
        if inpt.lower() == 'y' or inpt.lower() == 'yes':
            print(f"Saving trained model to {model_save_path}...")
            torch.save(net.state_dict(), model_save_path)
            print("Model saved.")

    # --- Test with a few generated images ---
    print("\n--- Testing with a few generated two-digit images ---")
    for n, (x, y) in enumerate(test_data):
        if n >= 5: # Show 5 examples
            break
        # Get the first image and its labels from the batch
        img_to_show = x[0]
        true_label1 = y[0, 0].item()
        true_label2 = y[0, 1].item()

        # Get predictions
        output_digit1, output_digit2 = net.forward(img_to_show.unsqueeze(0)) # Add batch dimension
        predicted_digit1 = torch.argmax(output_digit1).item()
        predicted_digit2 = torch.argmax(output_digit2).item()

        plt.figure(figsize=(4, 2))
        # Remove the channel dimension for imshow (if it's 1xHxC or CxHxW)
        plt.imshow(img_to_show.squeeze().numpy(), cmap='gray')
        plt.title(f"True: {true_label1}{true_label2}, Predicted: {predicted_digit1}{predicted_digit2}")
        plt.axis('off')
        plt.show()

    # --- User training/testing with custom images (simplified) ---
    # This section is adapted. It assumes the user provides a single 28x56 image
    # and labels it with two digits.
    print("\n--- User testing/training with custom two-digit images (experimental) ---")
    inpt = input("Do you want to test/train with custom two-digit images? (y/n) ").strip()
    if inpt.lower() == 'y' or inpt.lower() == 'yes':
        more_custom_training = True
        newtrain_custom_images = []
        newtrain_custom_labels = []

        # Placeholder for custom image path handling
        # In a real scenario, you'd need a robust way to load user-provided 28x56 images
        print("Please place your 28x56 grayscale PNG images in the same directory.")
        print("Name them like 'custom_0.png', 'custom_1.png', etc.")

        custom_image_files = sorted(glob.glob('custom_*.png'))
        if not custom_image_files:
            print("No custom images found. Skipping custom training/testing.")
            more_custom_training = False

        for i, img_path in enumerate(custom_image_files):
            try:
                image = Image.open(img_path).convert('L') # Ensure grayscale
                # Resize to 28x56 if not already
                if image.size != (56, 28):
                    print(f"Warning: Resizing {img_path} from {image.size} to 56x28.")
                    image = image.resize((56, 28), Image.LANCZOS)

                transform = transforms.Compose([transforms.ToTensor()])
                image_tensor = transform(image)
                input_batch = image_tensor.unsqueeze(0) # Add batch dimension

                output_digit1, output_digit2 = net.forward(input_batch)
                predict_digit1 = torch.argmax(output_digit1).item()
                predict_digit2 = torch.argmax(output_digit2).item()

                plt.figure(figsize=(4, 2))
                plt.imshow(image, cmap='gray')
                plt.title(f"Prediction: {predict_digit1}{predict_digit2} (from {os.path.basename(img_path)})")
                plt.axis('off')
                plt.show()

                user_input_label = input(f"What should the correct two-digit number be for {os.path.basename(img_path)}? (e.g., '42' or 'skip') ").strip()
                if len(user_input_label) == 2 and user_input_label.isdigit():
                    newtrain_custom_images.append(image_tensor)
                    newtrain_custom_labels.append(torch.tensor([int(user_input_label[0]), int(user_input_label[1])], dtype=torch.long))
                else:
                    print("Invalid input. Skipping this image for training.")

            except FileNotFoundError:
                print(f"Error: Custom image {img_path} not found. Skipping.")
            except Exception as e:
                print(f"An error occurred processing {img_path}: {e}. Skipping.")

        if newtrain_custom_images:
            print("\n--- Performing additional training with custom images ---")
            # Create a custom DataLoader for the new user-provided data
            custom_dataset = list(zip(newtrain_custom_images, newtrain_custom_labels))
            custom_dataloader = DataLoader(custom_dataset, batch_size=len(newtrain_custom_images), shuffle=True)

            # Perform a few more training steps on custom data
            for epoch in range(2): # Train for 2 epochs on custom data
                for (x, y) in custom_dataloader:
                    net.zero_grad()
                    output_digit1, output_digit2 = net.forward(x)
                    loss1 = F.nll_loss(output_digit1, y[:, 0])
                    loss2 = F.nll_loss(output_digit2, y[:, 1])
                    loss = loss1 + loss2
                    loss.backward()
                    optimizer.step()
                print(f"Custom training epoch {epoch+1} complete.")

            print("After custom training, accuracy on test data:", evaluate(test_data, net))

            inpt = input("Save model after custom training? (y/n) ").strip()
            if inpt.lower() == 'y' or inpt.lower() == 'yes':
                print(f"Saving trained model to {model_save_path}...")
                torch.save(net.state_dict(), model_save_path)
                print("Model saved.")
        else:
            print("No valid custom images provided for training.")


if __name__ == "__main__":
    main()

