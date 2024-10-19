import torch
from torch.utils.data import DataLoader
import argparse

from model import EncoderModel, tokenize_and_pad, EncoderDecoderModel
from qr_dataset import QRDataSet
from training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train QR Code Generator")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--embed_size", type=int, default=1024, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=4816, help="Hidden size")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize model with arguments
    model = EncoderDecoderModel(embed_size=args.embed_size,
                                hidden_size=args.hidden_size,
                                num_encoder_layers=args.num_encoder_layers,
                                num_heads=args.num_heads)

    # Create datasets
    train_dataset = QRDataSet(size=args.train_size)


    def collate_fn(batch):
        strings, qr_tensors = zip(*batch)
        tokens, attention_mask = tokenize_and_pad(strings, model.max_length, model.pad_token_id)
        return tokens, attention_mask, torch.stack(qr_tensors)


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Train the model
    trained_model = train_model(model, train_loader, args.num_epochs, args.learning_rate, args.device)

    # Save the trained model
    torch.save(trained_model.state_dict(), "qr_generator_model.pth")
    print("Model saved as 'qr_generator_model.pth'")

    # Test the model
    model.eval()
    test_strings = ["Hello, World!", "Python is great", "QR Codes"]
    tokens, attention_mask = tokenize_and_pad(test_strings, model.max_length, model.pad_token_id)
    tokens = tokens.to(args.device)
    attention_mask = attention_mask.to(args.device)

    with torch.no_grad():
        output = model(tokens, attention_mask)

    for i, test_string in enumerate(test_strings):
        print(f"Test Input: {test_string}")
        print(f"Output shape: {output[i].shape}")
        print(f"Number of 1s: {output[i].sum().item()}")
        print(f"Number of 0s: {(output[i] == 0).sum().item()}")
        print()
