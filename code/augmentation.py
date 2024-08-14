import numpy as np
from PIL import Image
import albumentations as A
import torch.nn as nn
from annotation_generator import AnnotationGenerator
from torchvision import transforms
import random


class ImageFolder(nn.Module):
    def __init__(self, img_size, num_images):
        super(ImageFolder, self).__init__()
        self.size = img_size
        self.to_tensor = transforms.ToTensor()
        ag = AnnotationGenerator()
        self.images = ag.get_matching_images(num_images)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        transform = self.random_transformation()
        augmented_pair = []
        for image in self.images[i]:
            image = Image.open(image)
            augmented = transform(image=np.array(image))
            image = augmented["image"]
            augmented_pair.append(image)

        # to tensor & normalise
        augmented_pair = [self.to_tensor(image) for image in augmented_pair]
        augmented_pair = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image) for image in augmented_pair]

        return augmented_pair
    
    def random_transformation(self):
        s1, s2 = int(self.size*3.14), self.size
        do_transformations = [A.CenterCrop(height=s1, width=s1), A.Resize(width=s2, height=s2)]
        for transformation in [ A.HorizontalFlip(p=1), A.VerticalFlip(p=1)]:
            if random.randint(0,1):
                do_transformations.append(transformation)
        if random.randint(0,1):
            r, g, b = [random.randint(-20,20) for _ in range(3)]
            do_transformations.append(A.RGBShift(p=1, r_shift_limit=(r,r), g_shift_limit=(g,g), b_shift_limit=(b,b)))
        if random.randint(0,1):
            b, c, s = [random.randint(7,10)/10 for _ in range(3)]
            h = random.randint(0,5)/10
            do_transformations.append(A.ColorJitter(p=1, brightness=(b,b), contrast=(c,c), saturation=(s,s), hue=(h,h)))
        if random.randint(0,1):
            l = random.randrange(1,4,2)
            do_transformations.append(A.Blur(blur_limit=(l,l), p=1))
        return A.Compose(do_transformations)
