import os
import random
from glob import glob
from PIL import Image
from annotation_generator import AnnotationGenerator

CROPPED_DIR, MATCHING_DIR = '../images/cropped', '../images/matching'


class ImageLoader(AnnotationGenerator):
  def __init__(self):
    super().__init__()
    self.default_folder_size = 600
    return
  
  def get_matching_images(self, total_amount=None):
    current_amount = self.num_matching_images()
    if total_amount == None: total_amount = current_amount
    if total_amount > current_amount:
      self.set_num_folders(total_amount // self.default_folder_size + 1)
      self.set_folder_sizes(self.default_folder_size)
    matching_images = [] # list of path pairs
    matching_dirs = [dir for dir in os.listdir(MATCHING_DIR) if os.path.isdir(f"{MATCHING_DIR}/{dir}")]
    while len(matching_images) < total_amount:
      selected_dir = matching_dirs.pop(random.randint(0, len(matching_dirs)-1))
      len_dir = len(os.listdir(f'{MATCHING_DIR}/{selected_dir}/clean'))
      imgs_left = total_amount - len(matching_images)
      selected_images = random.sample(glob(f'{MATCHING_DIR}/{selected_dir}/clean/*.jpg'), min(len_dir, imgs_left))
      for clean_img in selected_images:
        pair = [clean_img]
        annot_img = clean_img.replace("clean", "annotated")
        if os.path.exists(annot_img):
          pair.append(annot_img)
          matching_images.append(pair)
          for img_path in pair:
            try:
              Image.open(img_path).load()
            except Exception as e:
              print(f"Corrupted image: {img_path}")
              for img in pair:
                if os.path.exists(img): os.remove(img)
        else:
          print(f"Missing annotated image for {clean_img}")
          os.remove(clean_img)
    return matching_images
  
  def num_matching_images(self):
    total = 0
    for type in ["clean", "annotated"]:
      total += sum([len(os.listdir(f'{MATCHING_DIR}/{dir}/{type}')) for dir in os.listdir(MATCHING_DIR) if os.path.isdir(f'{MATCHING_DIR}/{dir}')])
    return total
  
  def set_folder_size(self, new_num_images):
    for folder_type in ["clean", "annotated"]:
      folder = f'{MATCHING_DIR}/matching {self.save_dir}/{folder_type}'
      # delete images in folder
      for img_path in os.listdir(folder):
        os.remove(f'{folder}/{img_path}')
    # add images to folder
    self.save_matching(new_num_images)
    return
  
  def get_folder_size(self, folder_num=None):
    if folder_num == None : folder_num = self.save_dir
    matching_folder = f'{MATCHING_DIR}/matching {folder_num}'
    clean_size, annot_size = 0, 0
    if os.path.exists(f'{matching_folder}/clean'):
      clean_size = len(os.listdir(f'{matching_folder}/clean'))
    if os.path.exists(f'{matching_folder}/annotated'):
      annot_size = len(os.listdir(f'{matching_folder}/annotated'))
    size = max(clean_size, annot_size)
    return size
  
  def set_num_folders(self, num_folders):
    # remove extra folders
    for i in range(num_folders, len(os.listdir(MATCHING_DIR))):
      folder = f'{MATCHING_DIR}/matching {i}'
      if os.path.exists(folder):
        # empty folder before removing it
        for item in os.listdir(folder):
          if os.path.isdir(f'{folder}/{item}'):
            for file in os.listdir(f'{folder}/{item}'):
              os.remove(f'{folder}/{item}/{file}')
            os.rmdir(f'{folder}/{item}')
          else:
            os.remove(f'{folder}/{item}')
        os.rmdir(folder)
    # add missing folders
    for i in range(len(os.listdir(MATCHING_DIR)), num_folders):
      folder = f'{MATCHING_DIR}/matching {i}'
      if not os.path.exists(folder):
        os.makedirs(folder)
    return
  
  def set_folder_sizes(self, folder_size):
    for folder_num, _ in enumerate(os.listdir(MATCHING_DIR)):
      self.switch_save_folder(folder_num)
      current_folder_size = self.get_folder_size()
      if current_folder_size != folder_size:
        print(f"Folder {folder_num} size: {current_folder_size}")
        self.set_folder_size(folder_size)
        print(f"Set to size: {self.get_folder_size()}")
    return


if __name__ == "__main__":
  il = ImageLoader()
  num_folders, folder_size = 210, 600
  print("Num folders:", num_folders)
  print("Folder size:", folder_size)
  il.set_num_folders(num_folders)
  il.set_folder_sizes(folder_size)
  print("Matching pairs:", il.num_matching_images())
  print("Clean images:", il.num_cropped_images())