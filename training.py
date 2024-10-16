import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb


def train_model(model, train_loader, num_epochs, learning_rate, device, log_interval=100):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb for logging
    wandb.init(project="qr-code-generator", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size
    })

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for tokens, attention_mask, target_qr in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            target_qr = target_qr.to(device)

            optimizer.zero_grad()
            output = model(tokens, attention_mask)
            loss = criterion(output, target_qr)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

            # Log additional metrics and images at regular intervals
            if global_step % log_interval == 0:
                # Threshold the output to get binary values
                pred_binary = (output > 0.5).float()

                # Calculate pixel accuracy
                correct_pixels = (pred_binary == target_qr).float().sum()
                total_pixels = torch.numel(target_qr)
                pixel_accuracy = (correct_pixels / total_pixels) * 100  # percentage

                # Log images
                pred_image = pred_binary[0].cpu().detach().view(21, 21).numpy()
                target_image = target_qr[0].cpu().view(21, 21).numpy()

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(pred_image, cmap='binary')
                ax1.set_title('Predicted')
                ax1.axis('off')
                ax2.imshow(target_image, cmap='binary')
                ax2.set_title('Target')
                ax2.axis('off')
                plt.tight_layout()

                # Log to wandb
                wandb.log({
                    "qr_codes": wandb.Image(fig, caption="Predicted vs Target"),
                    "pixel_accuracy": pixel_accuracy.item(),
                    "train_loss": loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                }, step=global_step)
                plt.close(fig)

        train_loss /= len(train_loader)
        wandb.log({"epoch_train_loss": train_loss}, step=global_step)

        print("-" * 50)

    wandb.finish()
    return model
