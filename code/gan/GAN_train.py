import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from GAN_model import Discriminator, initialize_weights
from UNET_model import UNET
from augmentation import ImageFolder
import os


if __name__ == "__main__":
    # device = cuda | mips | cpu
    if torch.backends.cuda.is_built(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    x = torch.ones(1, device=device)
    print (x)

    LOAD_MODEL = True
    D_LEARNING_RATE, G_LEARNING_RATE = 2e-4, 3e-4 # 2, 3
    BATCH_SIZE, NUM_BATCHES, NUM_EPOCHS = 8, 500, 50
    IMAGE_SIZE, CHANNELS_IMG = 128, 3
    NUM_WORKERS = 8
    RESULTS_DIR = f"results/size {IMAGE_SIZE}"
    if not LOAD_MODEL: RESULTS_DIR = f"results/new {IMAGE_SIZE}"
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    transformed_dataset = ImageFolder(IMAGE_SIZE, BATCH_SIZE*NUM_BATCHES)
    train_dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    gen = UNET(3, 3).to(device)
    disc = Discriminator(CHANNELS_IMG, IMAGE_SIZE).to(device)
    if LOAD_MODEL:
        gen.load_state_dict(torch.load(f"{RESULTS_DIR}/gen_checkpoint.pth.tar")["state_dict"])
        disc.load_state_dict(torch.load(f"{RESULTS_DIR}/disc_checkpoint.pth.tar")["state_dict"])
    else:
        initialize_weights(disc)
    opt_gen = optim.Adam(gen.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    prev_logs = os.listdir('code/logs')
    write_num = len(prev_logs)
    for i in range(len(prev_logs)):
        if "log "+str(i) not in prev_logs:
            write_num = i
            break
    writer = SummaryWriter(f"code/logs/log {write_num}")
    step = 0

    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, annotated) in enumerate(train_dataloader):
            real = real.to(device)
            annotated = annotated.to(device)
            fake = gen(annotated)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses
            if batch_idx % 100 == 0:
                step += 1
                print(
                        f"  Step: {step} \t \
                        Epoch [{epoch+1}/{NUM_EPOCHS}], Batch {str(batch_idx).zfill(3)}/{len(train_dataloader)} \t \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                # Print to tensorboard
                with torch.no_grad():
                    EXAMPLES = min(768//IMAGE_SIZE, BATCH_SIZE)
                    img_grid_real = torchvision.utils.make_grid(real[:EXAMPLES], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:EXAMPLES], normalize=True)
                    img_grid_annotated = torchvision.utils.make_grid(annotated[:EXAMPLES], normalize=True)

                    writer.add_image(f"real {write_num}", img_grid_real, global_step=step)
                    writer.add_image(f"fake {write_num}", img_grid_fake, global_step=step)
                    writer.add_image(f"annotated {write_num}", img_grid_annotated, global_step=step)

                # save checkpoint
                gen_checkpoint = {"state_dict": gen.state_dict(), "optimizer": opt_gen.state_dict()}
                torch.save(gen_checkpoint, f"{RESULTS_DIR}/gen_checkpoint.pth.tar")
                disc_checkpoint = {"state_dict": disc.state_dict(), "optimizer": opt_disc.state_dict()}
                torch.save(disc_checkpoint, f"{RESULTS_DIR}/disc_checkpoint.pth.tar")