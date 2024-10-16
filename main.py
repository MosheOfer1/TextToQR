import torch
import numpy as np
from torch.utils.data import DataLoader

from model import Model, tokenize_and_pad
from qr_dataset import QRDataSet
from training import train_model


def image_to_tensor(image):
    """
    Convert a 21x21 image to a 1D tensor of size 441.

    Args:
    image (PIL.Image): Input 21x21 image

    Returns:
    torch.Tensor: 1D tensor of size 441
    """
    # Ensure the input is the correct size
    assert image.size == (21, 21), "Input image must be 21x21 pixels"

    # Convert the image to a numpy array
    img_array = np.array(image)

    # Ensure the image is in grayscale
    if len(img_array.shape) == 3:
        img_array = img_array.mean(axis=2)

    # Normalize the values to 0-1
    img_array = img_array / 255.0

    # Convert to tensor and reshape to 1D
    return torch.from_numpy(img_array).float().view(-1)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SIZE = 10000
    VAL_SIZE = 2000
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = Model()

    # Create datasets
    train_dataset = QRDataSet(size=TRAIN_SIZE)

    def collate_fn(batch):
        strings, qr_tensors = zip(*batch)
        tokens, attention_mask = tokenize_and_pad(strings, model.max_length, model.pad_token_id)
        return tokens, attention_mask, torch.stack(qr_tensors)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Train the model
    trained_model = train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)

    # Save the trained model
    torch.save(trained_model.state_dict(), "qr_generator_model.pth")
    print("Model saved as 'qr_generator_model.pth'")

    # Test the model
    model.eval()
    test_strings = ["Hello, World!", "Python is great", "QR Codes"]
    tokens, attention_mask = tokenize_and_pad(test_strings, model.max_length, model.pad_token_id)
    tokens = tokens.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    with torch.no_grad():
        output = model(tokens, attention_mask)

    for i, test_string in enumerate(test_strings):
        print(f"Test Input: {test_string}")
        print(f"Output shape: {output[i].shape}")
        print(f"Number of 1s: {output[i].sum().item()}")
        print(f"Number of 0s: {(output[i] == 0).sum().item()}")
        print()
