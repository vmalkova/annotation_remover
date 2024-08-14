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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TO_LOAD = 1   # 0 from scratch
IMAGE_SIZE, CHANNELS_IMG = 256, 3
BATCH_SIZE, NUM_BATCHES, NUM_EPOCHS = 100, 300, 100
NUM_WORKERS, LEARNING_RATE = 12, 1e-4
STEP_SIZE = NUM_BATCHES//20
RESULTS_DIR = f"../results/size{IMAGE_SIZE}_{BATCH_SIZE}"
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
if not MODEL_TO_LOAD: MODEL_TO_LOAD = len(os.listdir(RESULTS_DIR)) + 1
CHECKPOINT_PATH = f"{RESULTS_DIR}/unet_checkpoint{str(MODEL_TO_LOAD).zfill(3)}.pth.tar" 
LOG_DIR = f"log{IMAGE_SIZE}_{BATCH_SIZE}_{str(MODEL_TO_LOAD).zfill(3)}"
if LOG_DIR in os.listdir("logs"): os.system(f"rm -r logs/{LOG_DIR}")
os.makedirs(f"logs/{LOG_DIR}")
torch.cuda.empty_cache()

def print_to_tensorboard(step):
    EXAMPLES = 8
    with torch.no_grad():
        for real, annotated in var_dataloader:
            real = real.to(DEVICE)
            annotated = annotated.to(DEVICE)
            prediction = model(annotated)
            img_grid_real = torchvision.utils.make_grid(real[:EXAMPLES], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(prediction[:EXAMPLES], normalize=True)
            img_grid_annotated = torchvision.utils.make_grid(annotated[:EXAMPLES], normalize=True)
            writer.add_image(f"real", img_grid_real, global_step=step)
            writer.add_image(f"fake", img_grid_fake, global_step=step)
            writer.add_image(f"annotated", img_grid_annotated, global_step=step)
    return

def train(model, lr_scheduler, optimizer, loss_fn, scaler):
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_dataloader)
        for batch_idx, (real, annotated) in enumerate(train_dataloader):
            optimizer.zero_grad()
            annotated = annotated.to(DEVICE)
            real = real.to(DEVICE)
            # forward
            with torch.cuda.amp.autocast():
                prediction = model(annotated)
                loss = loss_fn(prediction, real)
            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step(loss)
            # Print losses
            if batch_idx % STEP_SIZE == 0:
                pbar.set_postfix(loss = lr_scheduler.get_last_lr()[0])
                pbar.update(STEP_SIZE)
        print_to_tensorboard(step = epoch*NUM_BATCHES)
        gen_checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(gen_checkpoint, CHECKPOINT_PATH)
    return


if __name__ == "__main__":
    transformed_dataset = ImageFolder(IMAGE_SIZE, BATCH_SIZE*NUM_BATCHES)
    train_dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    var_dataset = ImageFolder(IMAGE_SIZE, BATCH_SIZE)
    var_dataloader = DataLoader(var_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    model = UNET(3, 3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    # if checkpoint path exists, load model and optimizer
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"])
        optimizer.load_state_dict(torch.load(CHECKPOINT_PATH)["optimizer"])
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(f"logs/{LOG_DIR}")
    train(model, lr_scheduler, optimizer, loss_fn, scaler)