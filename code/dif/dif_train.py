import torch
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as transforms
from accelerate import Accelerator
from diffusers import DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from PIL import Image
from dif_model import ButterflyModel
from dataclasses import dataclass
from augmentation import ImageFolder


NUM_BATCHES = 20
NUM_WORKERS = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 6
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 100 # 500
    save_image_epochs = 2
    save_model_epochs = 4
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'result'  # the model name locally and on the HF Hub
    load_model = True
    seed = 0
    dataset_name  = "rocks"


def print_to_tensorboard(step, prediction):
    EXAMPLES = min(768//config.image_size,8)
    with torch.no_grad():
        for real, annotated in train_dataloader:
            real = real.to(DEVICE)
            annotated = annotated.to(DEVICE)
            print(prediction.type())

            # img_grid_real = torchvision.utils.make_grid(real[:EXAMPLES], normalize=True)
            # img_grid_fake = torchvision.utils.make_grid(prediction[:EXAMPLES], normalize=True)
            # img_grid_annotated = torchvision.utils.make_grid(annotated[:EXAMPLES], normalize=True)

            # writer.add_image(f"real", img_grid_real, global_step=step)
            # writer.add_image(f"fake", img_grid_fake, global_step=step)
            # writer.add_image(f"annotated", img_grid_annotated, global_step=step)
    return

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.Generator(device=DEVICE).manual_seed(config.seed)
    ).images
    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    print_to_tensorboard(step=epoch, prediction=images)

def train_loop(config, noise_scheduler, train_dataloader):
    global model, optimizer, lr_scheduler
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            print(f"make output dir {config.output_dir}")
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # train
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(train_dataloader)
        progress_bar.set_description(f"Epoch {epoch}")
        for i, (clean, _) in enumerate(train_dataloader):
            clean_images = clean.to(DEVICE)
            noise = torch.randn(clean_images.shape).to(DEVICE)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=DEVICE).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(DEVICE)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            STEP_SIZE = NUM_BATCHES//10
            if i % STEP_SIZE == 0:
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": i}
                # accelerator.log(logs, step=epoch*NUM_BATCHES+i)
                progress_bar.set_postfix(**logs)
                progress_bar.update(STEP_SIZE)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)


if __name__ == "__main__":
    config = TrainingConfig()
    dataset = ImageFolder(config.image_size, config.train_batch_size*NUM_BATCHES)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, num_workers=NUM_WORKERS, shuffle=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = ButterflyModel(config)
    if config.load_model:
        model = model.from_pretrained(f"{config.output_dir}/unet")
    model = ButterflyModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    writer = SummaryWriter(f"{config.output_dir}/my_logs")
    train_loop(config, noise_scheduler, train_dataloader)