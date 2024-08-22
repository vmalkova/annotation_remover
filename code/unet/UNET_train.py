import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from augmentation import ImageFolder
from UNET_model import UNET
from tqdm import tqdm


MODEL_TO_LOAD = -1   # 0 from scratch, -1 last model
IMAGE_SIZE, CHANNELS_IMG = 256, 3
BATCH_SIZE, NUM_BATCHES, NUM_EPOCHS = 8,100, 1000
NUM_WORKERS, LEARNING_RATE, DECAY_RATE = 12, 3e-5, 0.96
TRAIN_SIZE, VAL_SIZE = int(0.8*BATCH_SIZE*NUM_BATCHES), int(0.2*BATCH_SIZE*NUM_BATCHES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = f"../results/size{IMAGE_SIZE}_{BATCH_SIZE}"
if not os.path.exists(results_dir): os.makedirs(results_dir)
models_saved = [int(x.split("unet_checkpoint")[1][:4]) for x in os.listdir(results_dir) if "unet_checkpoint" in x]
model_to_save = 1 if models_saved==[] else sorted(models_saved)[-1]+1
save_path = f"{results_dir}/unet_checkpoint{str(model_to_save).zfill(4)}.pth.tar" 
log_dir = f"logs/log{IMAGE_SIZE}_{BATCH_SIZE}_{str(model_to_save).zfill(4)}"
if os.path.exists(log_dir): os.system(f"rm -r {log_dir}/")
else: os.makedirs(log_dir)


def print_to_tensorboard(epoch, lr):
    examples = 8
    step_size = VAL_SIZE//20
    with torch.no_grad():
        pbar = tqdm(val_dataloader)
        pbar.set_description(f"Validation {epoch}")
        for batch_idx, (real, annotated) in enumerate(val_dataloader):
            with torch.no_grad():
                real = real.to(device)
                annotated = annotated.to(device)
            with torch.cuda.amp.autocast():
                prediction = model(annotated)
                prediction = torch.sigmoid(prediction)
                loss = loss_fn(prediction, real)
            if batch_idx % step_size == 0:
                pbar.set_postfix(loss = loss.item())
                pbar.update(step_size)
            # Save first batch of images
            if batch_idx == 0:
                img_grid_real = torchvision.utils.make_grid(real[:examples], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(prediction[:examples], normalize=True)
                img_grid_annotated = torchvision.utils.make_grid(annotated[:examples], normalize=True)
                writer.add_image(f"real", img_grid_real, global_step=epoch)
                writer.add_image(f"fake", img_grid_fake, global_step=epoch)
                writer.add_image(f"annotated", img_grid_annotated, global_step=epoch)
                writer.add_scalar("loss", loss.item(), global_step=epoch)
                writer.add_scalar("lr", lr, global_step=epoch)
        torch.cuda.empty_cache()
    return

def train(model, optimizer, loss_fn, scaler):
    step_size = TRAIN_SIZE//20
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch {epoch}")
        loss = 0
        for batch_idx, (real, annotated) in enumerate(train_dataloader):
            with torch.no_grad():
                real = real.to(device)
                annotated = annotated.to(device)
            optimizer.zero_grad()
            # forward
            with torch.cuda.amp.autocast():
                prediction = model(annotated)
                prediction = torch.sigmoid(prediction)
                loss = loss_fn(prediction, real)
            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Print losses
            if batch_idx % step_size == 0:
                pbar.set_postfix(loss = loss.item())
                pbar.update(step_size)
        torch.cuda.empty_cache()
        lr_scheduler.step(loss.item())
        if epoch % 50 == 0:
            lr = lr_scheduler.get_last_lr()[0]
            print_to_tensorboard(epoch, lr)
            gen_checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(gen_checkpoint, save_path)
    return


if __name__ == "__main__":
    transformed_dataset = ImageFolder(IMAGE_SIZE, TRAIN_SIZE)
    train_dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_dataset = ImageFolder(IMAGE_SIZE, VAL_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    model = UNET(3, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    if not (MODEL_TO_LOAD == 0 or model_to_save == 1):
        if MODEL_TO_LOAD == -1: load_num = model_to_save - 1
        else: load_num = MODEL_TO_LOAD
        load_path = f"{results_dir}/unet_checkpoint{str(load_num).zfill(4)}.pth.tar"
        model.load_state_dict(torch.load(load_path)["state_dict"])
        optimizer.load_state_dict(torch.load(load_path)["optimizer"])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir)
    train(model, optimizer, loss_fn, scaler)