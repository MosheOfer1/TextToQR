import torch
from torch.utils.data import Dataset
import qrcode
import numpy as np
import random
import string


class QRDataSet(Dataset):
    def __init__(self, size=1000, min_length=1, max_length=14):
        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate a random string
        length = random.randint(self.min_length, self.max_length)
        random_string = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))

        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=1, border=0)
        qr.add_data(random_string)
        qr.make(fit=True)

        # Get the QR code as a numpy array
        qr_array = np.array(qr.get_matrix(), dtype=np.float32)

        # Ensure the QR code is 21x21
        if qr_array.shape != (21, 21):
            raise ValueError(f"Generated QR code is not 21x21: {qr_array.shape}")

        # Flatten the array and convert to tensor
        qr_tensor = torch.tensor(qr_array.flatten(), dtype=torch.float32)

        return random_string, qr_tensor


# Example usage:
if __name__ == "__main__":
    dataset = QRDataSet(size=5, max_length=10)

    for i in range(len(dataset)):
        original_string, qr_tensor = dataset[i]
        print(f"Item {i}:")
        print(f"Original string: {original_string}")
        print(f"Tensor shape: {qr_tensor.shape}")
        print(f"Tensor: {qr_tensor}")
        print(f"Tensor sum: {qr_tensor.sum()}")  # Quick check to ensure we have both black and white pixels
        print()
