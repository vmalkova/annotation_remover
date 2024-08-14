import os
from glob import glob
import random
from drawing import Draw
from PIL import Image

CROPPED_DIR, MATCHING_DIR = '../images/cropped', '../images/matching'


class AnnotationGenerator():
  def __init__(self):
    self.save_dir = 0
    return

  def annotate(self, img=None, path=None):
    if path == None: drawing = Draw(img=img)
    else: drawing = Draw(path=path)
    space = drawing.img_space
    space.free_all()

    # draw measurement
    length = random.randint(50, 400)
    x, y = random.randint(150, drawing.w-length-50), random.choice([120, drawing.h - 50])
    space.fill((x, y), max(170, length+20), 250)
    drawing.add_measurement(x, y, length)

    # draw label in the corner
    if random.randint(1,10)>1:
      top = random.getrandbits(1)
      left = x*2 > drawing.w-length # measurement and label are on oposite sides
      d = (drawing.size+3)*30
      space.fill(([drawing.w-150, 150][left], [drawing.h-150, 150][top]), d, d)
      drawing.add_corner_label(top, left)

    # number of labels
    num_labels = random.randint(5,6)
    num_text = random.randint(0,num_labels)
    num_arrows = num_labels-num_text

    # plain text
    for _ in range(num_text):
      d = (drawing.size+4)*45
      x, y = space.get_available(d, d, run_fill=True)
      if (x, y) == (0, 0) : break
      drawing.add_text(x, y, big=False, size=random.randint(1,5), border=False)

    # text with arrows
    for _ in range(num_arrows):
      size, length = random.randint(1,5), random.randint(50,200)
      d = (drawing.size+7)*15
      x, y = space.get_available(d+length*2, d+length*2, run_fill=True)
      if (x, y) == (0, 0) : length = int(length*0.7) ; continue
      drawing.add_text_arrow(x, y, length=length, size=size)

    return drawing.image
  
  def get_cropped_images(self, total_amount):
    cropped_images = [] # list of paths
    cropped_dirs = [dir for dir in os.listdir(CROPPED_DIR) if os.path.isdir(f"{CROPPED_DIR}/{dir}")]
    while len(cropped_images) < total_amount:
      selected_dir = cropped_dirs.pop(random.randint(0, len(cropped_dirs)-1))
      len_dir = len(os.listdir(f'{CROPPED_DIR}/{selected_dir}'))
      imgs_left = total_amount - len(cropped_images)
      selected_images = random.sample(glob(f'{CROPPED_DIR}/{selected_dir}/*.jpg'), min(len_dir, imgs_left))
      cropped_images.extend(selected_images)
    return cropped_images
  
  def get_matching_images(self, total_amount=None):
    if total_amount == None: total_amount = self.num_matching_images()
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
                os.remove(img)
        else:
          print(f"Missing annotated image for {clean_img}")
          # delete clean image if annotated image is missing
          os.remove(clean_img)
    return matching_images
  
  def num_cropped_images(self):
    return sum([len(os.listdir(f'{CROPPED_DIR}/{dir}')) for dir in os.listdir(CROPPED_DIR) if os.path.isdir(f'{CROPPED_DIR}/{dir}')])
  
  def num_matching_images(self):
    return sum([len(os.listdir(f'{MATCHING_DIR}/{dir}/clean')) for dir in os.listdir(MATCHING_DIR) if os.path.isdir(f'{MATCHING_DIR}/{dir}')])
  
  def save_image(self, drawing:Draw, folder_type):
    folder = f'{MATCHING_DIR}/matching {self.save_dir}/{folder_type}'
    if not os.path.exists(folder):
      os.makedirs(folder)
    for i in range(len(os.listdir(folder))+1):
      str_i = str(i).zfill(3)
      img_name = f"{folder_type} {self.save_dir}_{str_i}.jpg"
      if os.path.exists(f"{folder}/{img_name}"): continue
      else:
        drawing.save(f"{folder}/{img_name}")
        return
    return

  def save_matching(self, num_images):
    for img_path in self.get_cropped_images(num_images):
      self.save_image(Draw(path=img_path), "clean")
      self.save_image(Draw(img=self.annotate(path=img_path)), "annotated")
    return
  
  def remove_matching(self, num_images):
    for folder_type in ["clean", "annotated"]:
      folder = f'{MATCHING_DIR}/matching {self.save_dir}/{folder_type}'
      if os.path.exists(folder):
        files = os.listdir(folder)
        files.sort()
        if files != []:
          for _ in range(num_images):
            file_to_remove = files.pop()
            os.remove(f'{folder}/{file_to_remove}')
    return

  def switch_save_folder(self, save_dir=None):
    if save_dir == None:
      matching_dirs = os.listdir(MATCHING_DIR)
      for i, _ in enumerate(matching_dirs):
        if f"matching {i}" not in matching_dirs:
          save_dir = i
          break
    self.save_dir = save_dir
    return
  
  def folder_size(self, folder_num=None):
    if folder_num == None : folder_num = self.save_dir
    folder = f'{MATCHING_DIR}/matching {folder_num}/clean'
    size = 0
    if os.path.exists(folder): size = len(os.listdir(folder))
    return size
  
  def set_num_folders(self, num_folders):
    # remove extra folders
    for i in range(num_folders, len(os.listdir(MATCHING_DIR))):
      if os.path.exists(f'{MATCHING_DIR}/matching {i}'):
        # empty folder before removing it
        for item in os.listdir(f'{MATCHING_DIR}/matching {i}'):
          if os.path.isdir(f'{MATCHING_DIR}/matching {i}/{item}'):
            for file in os.listdir(f'{MATCHING_DIR}/matching {i}/{item}'):
              os.remove(f'{MATCHING_DIR}/matching {i}/{item}/{file}')
            os.rmdir(f'{MATCHING_DIR}/matching {i}/{item}')
          else:
            os.remove(f'{MATCHING_DIR}/matching {i}/{item}')
        os.rmdir(f'{MATCHING_DIR}/matching {i}')
    # add missing folders
    for i in range(len(os.listdir(MATCHING_DIR)), num_folders):
      if not os.path.exists(f'{MATCHING_DIR}/matching {i}'):
        os.makedirs(f'{MATCHING_DIR}/matching {i}')
    return
  
  def set_folder_sizes(self, num_folders, folder_size):
    for folder_num in range(num_folders):
      self.switch_save_folder(folder_num)
      current_folder_size = self.folder_size()
      print(f"Folder {folder_num} size: {self.folder_size()}")
      if current_folder_size < folder_size:
        self.save_matching(folder_size-current_folder_size)
        print(f"Increased to size: {self.folder_size()}")
      elif current_folder_size > folder_size:
        self.remove_matching(current_folder_size-folder_size)
        print(f"Decreased to size: {self.folder_size()}")
    return


if __name__ == "__main__":
  ag = AnnotationGenerator()
  num_folders, folder_size = 30, 500
  print("Num folders:", num_folders)
  print("Folder size:", folder_size)
  ag.set_folder_sizes(num_folders, folder_size)
  print("Matching pairs:", ag.num_matching_images())
  print("Clean images:", ag.num_cropped_images())