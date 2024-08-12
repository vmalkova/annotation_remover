from PIL import Image, ImageDraw
import random
import string
import cv2
import re
from image_info import ImgInfo
import numpy


class Draw(ImgInfo):
    def __init__(self, img=None, path=None):
        super().__init__(img, path)

    def save(self, path):
        self.image.save(path)
        return

    def random_corner_text(self):
        text_choice = list(string.ascii_letters)
        text_choice.extend([str(i) for i in range(1,10)])
        text = random.choice(text_choice)
        text = random.choice([text, f"{text})", f"({text})"])
        return text

    def random_ltrs(self):
        ltrs = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(2,3)))
        return ltrs.capitalize()
    
    def random_dist(self):
        dist = str(random.choice([1,2,5])*10**(random.randint(1,3)))
        return dist + random.choice([" µm", " mm", "µm", "mm"])
    
    def add_lines(self, vertices, size=None):
        if size == None : size = self.size
        colour = self.img_colour.big_text
        draw = ImageDraw.Draw(self.image)
        for pair in vertices:
            draw.line(pair, fill=colour, width=size)
        return
    
    def add_text(self, x, y, text=None, big=False, shape=None, border=None, size=None):
        if text == None : text = self.random_ltrs()
        if shape == None : shape = self.shape
        if border == None : border = random.getrandbits(1)
        if size == None : size = self.size
        if big:
            text_colour = self.img_colour.big_text
            fill_colour = self.img_colour.big_fill
        else:
            text_colour = self.img_colour.small_text
            fill_colour = self.img_colour.small_fill
        len_text = self.img_font.text_length(text, size)
        yd = self.img_font.y_dif(size)
        x_left, y_top = int(x-len_text*0.3), y+yd-int(size*15)-30 #center text
        x, y = x, y-20-size*10
        # draw
        arr = numpy.array(self.image)
        h = (size+7)*5
        w = max(h, 20+int(len_text*0.3))
        if shape != "TEXT":
            if shape == "SQUARE" or h < w:
                arr = cv2.rectangle(arr, (x-w, y-h) , (x+w,y+h), fill_colour, -1)
                if border : arr = cv2.rectangle(arr, (x-w, y-h) , (x+w+1,y+h), text_colour, self.size)
            else:
                arr = cv2.circle(arr, (x, y), radius=h, color=fill_colour, thickness=-1)
                if border : arr = cv2.circle(arr, (x, y), radius=h, color=text_colour, thickness=self.size)
        self.image = Image.fromarray(arr)
        draw = ImageDraw.Draw(self.image)
        draw.text((x_left, y_top), text, font=self.img_font.font(size), fill=text_colour)
        return

    def add_corner_label(self, top, left):
        text = self.random_corner_text()
        border = random.getrandbits(1)
        size = self.size + 2
        d = int((size+6.5)*5)
        if border: d += size
        if self.shape == "CIRCLE" : d += int(size*0.7)+4 # circle is larger (so further from the edge)
        width, height = self.w, self.h
        x, y = [width-d, d][left], [height-d, d][top]
        if not left: text = re.sub('[()]', '', text) # if label is on the right, remove brackets
        self.add_text(x, y+20+size*10, text, big=True, shape=self.shape, border=border, size=size)
        return

    def add_measurement(self, x, y, length):
        text = self.random_dist()
        spikes = random.getrandbits(1)
        filled = random.getrandbits(1)
        text_colour = self.img_colour.big_text
        # work out location of elements
        size = self.size
        text_y, y = y, y - random.choice([0, 55+size*15])
        h, len_text = size, self.img_font.text_length(text, size)
        if spikes: h = min(18+size*2, size*14)
        if text_y != y:
            if spikes: text_y, y = text_y+h*2, y+h*2
            else: text_y, y = text_y+h*10, y+h*10
        vertices = [[(x,y),(x+length,y)]]
        if spikes: vertices.extend([[(x,y+h), (x,y-h)],[(x+length,y+h), (x+length,y-h)]])
        if filled: # draw background rectangle
            arr = numpy.array(self.image)
            fill_colour = self.img_colour.big_fill
            x1, y1 = min(x-20, x-(len_text-length)//2), y-h-20
            x2, y2 = x1+max(length+40, len_text), y+h+20
            arr = cv2.rectangle(arr, (x1, y1) , (x2, y2), fill_colour, -1)
            y1, y2 = text_y-(3+size)*15, text_y-size*2
            arr = cv2.rectangle(arr, (x1, y1), (x2, y2), fill_colour, -1)
            if random.getrandbits(1): # draw border
                if text_y != y : y1 = y-h-20
                else: y2 = y+h+20
                arr = cv2.rectangle(arr, (x1, y1) , (x2, y2), text_colour, size)
            self.image = Image.fromarray(arr)
        # draw measurement bar
        self.add_text(x+length//2, text_y, text, big=True, shape="TEXT", border=False, size=size)
        if not spikes: size += 4
        self.add_lines(vertices, size+2)
        return
    
    def add_text_arrow(self, x1, y1, length=None, size=None):
        if size == None : size = self.size
        if random.getrandbits(1): shape="SQUARE"; size=min(size, 3) # if filled, add square background
        else: shape="TEXT"
        text = self.random_ltrs()
        text_colour = self.img_colour.small_text
        # calculate start and end points
        if length == None:
            max_length = min(min(x1, self.w-x1), min(y1, self.h-y1))
            min_length - max_length//3
            if shape == "SQUARE": min_length = max(int(self.img_font.fontsize(size)*0.9)+15, min_length)
            length = random.randint(min_length, max_length)
        start_point = (x1, y1)
        xd = random.randint(0, length)
        end_point = (x1 + xd*random.choice([-1, 1]), y1 + (length-xd)*random.choice([-1, 1]))
        if end_point[1] < start_point[1]: y1 += 40+ size*20
        # draw
        self.add_text(x1, y1, text, shape=shape, border=False, size=size)
        arr = numpy.array(self.image)
        arr = cv2.arrowedLine(arr, start_point, end_point, text_colour, thickness = min(6, size + length//50))
        self.image = Image.fromarray(arr)
        return
