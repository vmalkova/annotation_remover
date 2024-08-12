from PIL import Image, ImageFont
import random


class ImgSpace():
  def __init__(self, width, height):
    self.width = width
    self.height = height
    self.free_positions = []

  def free_all(self):
    self.free_positions = [(x, y) for x in range(100, self.width, 60) for y in range(self.height-60, 100, -60)]
    return

  def fill(self, filled, obj_w, obj_h):
    obj_w, obj_h = obj_w//2, obj_h//2
    x1, x2 = filled[0]-obj_w, filled[0]+obj_w
    y1, y2 = filled[1]-obj_h, filled[1]+obj_h
    self.free_positions = [pos for pos in self.free_positions if (pos[0]<=x1 or pos[0]>=x2) and (pos[1]<=y1 or pos[1]>=y2)]
    return
  
  def get_available(self, obj_w, obj_h, min_x=None, max_x=None, min_y=None, max_y=None, run_fill=False):
    if min_x == None : min_x = obj_w
    if max_x == None : max_x = self.width - obj_w
    if min_y == None : min_y = obj_h
    if max_y == None : max_y = self.height - obj_h
    # remove the options that are too close to the min/max border
    options = [(x, y) for (x, y) in self.free_positions if x<=max_x and x>=min_x and y<=max_y and y>=min_y]
    
    # remove the options that are too close to taken spaces
    for (x, y) in options:
        for (dx, dy) in [(obj_w//2, 0), ((0-obj_w)//2, 0), (0, obj_h//2), (0, (0-obj_h)//2)]:
            if (x+dx, y+dy) not in self.free_positions:
              options.remove((x, y))
              break
    
    # choose a random option
    if options == [] : return (0, 0)
    free = random.choice(options)
    if run_fill: self.fill(free, obj_w, obj_h)
    return free

class ImgFont():
  def __init__(self):
    fonts = [ {'path': r'fonts/LiberationSans-Italic.ttf', 'size': 8, 'y_dif': 0},
              {'path': r'fonts/LiberationSerif-Italic.ttf', 'size': 7, 'y_dif': -1.5},
              {'path': r'fonts/LiberationSans-BoldItalic.ttf', 'size': 9, 'y_dif': 1},
              {'path': r'fonts/LiberationSerif-Bold.ttf', 'size': 8, 'y_dif': 0},
              {'path': r'fonts/LiberationSerif-Regular.ttf', 'size': 7, 'y_dif': -1},
              {'path': r'fonts/LiberationSerif-BoldItalic.ttf', 'size': 7, 'y_dif': -1}]
    self.font_d = random.choice(fonts)

  def text_length(self, text, size):
    length = 0
    for ltr in text:
        scale = 0.5
        if ltr in "LSXacehiµnprsz": scale = 0.75
        elif ltr in "0LTbghkoquvxyµ": scale = 0.9
        elif ltr in "123456789JPRSUXd": scale = 1
        elif ltr in "ABCDEFGHKNOQVYZmw": scale = 1.1
        elif ltr in "MW": scale = 1.3
        length += scale*self.fontsize(size)
    return int(length)
  
  def font(self, size):
    return ImageFont.truetype(self.font_d['path'], self.fontsize(size))

  def fontsize(self, size):
    return (size+2)*7000//(self.font_d['size']*100)
  
  def y_dif(self, size):
    return int(self.font_d['y_dif'] * (size+1))

class ImgColour():
  def __init__(self):
    self.colours = {"RED": (220, 0, 0), "YELLOW": (210, 210, 0), "BLACK": (0, 0, 0), "WHITE": (220, 220, 220)}
    self.small_text = self.random_colour()
    if self.small_text == self.colours["YELLOW"] : self.small_fill = self.colours["BLACK"]
    else: self.small_fill = self.random_colour(excluded=[self.small_text])
    self.big_text = self.random_colour(["WHITE", "BLACK"])
    self.big_fill = self.random_colour(["WHITE", "BLACK"], excluded=[self.big_text])

  def random_colour(self, included=None, excluded=[]):
    if included == None:
      included = [key for key, val in self.colours.items() if val not in excluded]
    else:
      included = [name for name in included if self.colours[name] not in excluded]
    return random.choice([self.colours[name] for name in included])

class ImgInfo():
  def __init__(self, img=None, path=None, width=1000, height=830):
    if path == None: self.image = img
    else: self.image = Image.open(path)
    self.w, self.h = width, height
    self.size = random.randint(1, 5)
    self.shape = random.choice(["SQUARE", "CIRCLE", "TEXT"])
    self.img_font = ImgFont()
    self.img_colour = ImgColour()
    self.img_space = ImgSpace(width, height)
    return