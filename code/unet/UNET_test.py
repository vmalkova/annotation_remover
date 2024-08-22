import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
import torch.nn as nn
from torchvision import transforms
from UNET_model import UNET


sizes = [path.split("size")[1] for path in os.listdir("results") if "size" in path]
image_size = int(sorted(sizes)[-1])
results_dir = f"results/size{image_size}"
if not os.path.exists(results_dir): models_saved = []
else: models_saved = [int(x.split("unet_checkpoint")[1][:4]) for x in os.listdir(results_dir) if "unet_checkpoint" in x]
model_to_load = 0 if models_saved==[] else sorted(models_saved)[-1]
device = torch.device("cpu")
load_img_dir, save_img_dir = "images/test/to_clean", "images/test/done"


class ImageFolder(nn.Module):
    def __init__(self, img_size):
        super(ImageFolder, self).__init__()
        self.size = img_size
        self.to_tensor = transforms.ToTensor()
        self.images = []
        for img in os.listdir(load_img_dir):
            img_path = f"{load_img_dir}/{img}"
            try:
                Image.open(img_path).load()
                if not any([img in name for name in os.listdir(save_img_dir)]):
                    self.images.append(img_path)
            except Exception:
                print(f"Corrupted image: {img}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        width, height = Image.open(self.images[i]).size
        resize = self.transformation(width, height)
        img = Image.open(self.images[i])
        new_img = resize(image=np.array(img))["image"]
        new_img = self.to_tensor(new_img)
        name = self.images[i].split("/")[-1]
        return new_img, name
    
    def transformation(self, width, height):
        s1, s2, actual_size = int(self.size*3.14), self.size, min(width, height)
        if actual_size < s1: s1 = actual_size
        transforms = [A.CenterCrop(height=s1, width=s1), A.Resize(width=s2, height=s2)]
        return A.Compose(transforms)


def run(model, dataloader):
    model.eval()
    with torch.no_grad():
        for _, (real, name) in enumerate(dataloader):
            real = real.to(device)
            prediction = model(real)
            prediction = torch.sigmoid(prediction)
            for i, img in enumerate(prediction):
                img = transforms.ToPILImage()(img)
                img.save(f"{save_img_dir}/clean {name[i]}")
    return


if __name__ == "__main__":
    if model_to_load == 0: print (f"Model (size {image_size}) doesn't exist"); exit()
    print (f"Model (size {image_size} v{model_to_load}) loaded")
    model = UNET(3, 3).to(device)
    load_path = f"{results_dir}/unet_checkpoint{str(model_to_load).zfill(4)}.pth.tar"
    model.load_state_dict(torch.load(load_path, map_location=device)["state_dict"])
    dataset = ImageFolder(image_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    run(model, dataloader)
    print ("Images saved")