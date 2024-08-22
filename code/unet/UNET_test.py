import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
import torch.nn as nn
from torchvision import transforms
from UNET_model import UNET


IMAGE_SIZE, BATCH_SIZE = 256, 16
results_dir = f"../results/size{IMAGE_SIZE}_{BATCH_SIZE}"
models_saved = [int(x.split("unet_checkpoint")[1][:4]) for x in os.listdir(results_dir) if "unet_checkpoint" in x]
model_to_load = 0 if models_saved==[] else sorted(models_saved)[-1]
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.backends.cuda.is_built(): device = torch.device("cuda")
else: device = torch.device("cpu")
load_img_dir, save_img_dir = "../images/test/to_clean", "../images/test/done"


class ImageFolder(nn.Module):
    def __init__(self, img_size):
        super(ImageFolder, self).__init__()
        self.size = img_size
        self.to_tensor = transforms.ToTensor()
        self.images = [f"{load_img_dir}/{x}" for x in os.listdir(load_img_dir)]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        width, height = Image.open(self.images[i]).size
        resize = self.transformation(width, height)
        img = Image.open(self.images[i])
        new_img = resize(image=np.array(img))["image"]
        return self.to_tensor(new_img)
    
    def transformation(self, width, height):
        s1, s2, actual_size = int(self.size*3.14), self.size, min(width, height)
        if actual_size < s1: s1 = actual_size
        transforms = [A.CenterCrop(height=s1, width=s1), A.Resize(width=s2, height=s2)]
        return A.Compose(transforms)


def run(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch_idx, real in enumerate(dataloader):
            real = real.to(device)
            prediction = model(real)
            prediction = torch.sigmoid(prediction)
            for i, img in enumerate(prediction):
                img = transforms.ToPILImage()(img)
                img.save(f"{save_img_dir}/{images[batch_idx*BATCH_SIZE+i]}")
    return


if __name__ == "__main__":
    if model_to_load == 0: print ("No model to load"); exit()
    print (f"Model {model_to_load} loaded")
    model = UNET(3, 3).to(device)
    load_path = f"{results_dir}/unet_checkpoint{str(model_to_load).zfill(4)}.pth.tar"
    model.load_state_dict(torch.load(load_path, map_location=device)["state_dict"])
    images = os.listdir(load_img_dir)
    dataset = ImageFolder(IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    run(model, dataloader)
    print ("Images saved")