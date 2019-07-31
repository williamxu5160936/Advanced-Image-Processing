import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

def ok_pressed(self):
    self.radius = float(self.radiusT.get())
    self.x_center = float(self.xT.get())
    self.y_center = float(self.yT.get())
    # print('radius: ' + str(self.radius) + 'Coordinates:  (' + str(self.x_center) + ', ' + str(self.y_center))
    self.circle = patches.Circle((float(self.x_center), float(self.y_center)), float(self.radius),
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
    if self.prevCirc is not None:
        self.prevCirc.set_visible(False)
    self.prevCirc = self.circle
    self.ax.add_patch(self.circle)
    plt.show()


def manual_mode(self):
    self.mode = 'Manual Mode'
    # print(self.mode)
    if(hasattr(StartPage, 'patches')):
        for c in self.patches:
            c.set_visible(False)
    plt.show()


def auto_mode(self):
    self.mode = 'Auto Mode'
    img = self.imgTemp
    g = (200.0 - 50.0) / (np.max(img) - np.min(img))
    o = 50.0 - g * np.min(img)
    imgA = (img * g + o).astype('uint8')

    img = cv2.medianBlur(imgA, 5)

    cimg = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=200, param2=30, minRadius=20, maxRadius=40)

    self.auto_circles = []
    self.patches = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            # cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            circle = patches.Circle((float(i[0]), float(i[1])), float(i[2]),
                                    linewidth=1,
                                    edgecolor='y',
                                    facecolor='none')
            self.auto_circles.append((float(i[0]), float(i[1]), float(i[2])))
            self.patches.append(circle)
            self.ax.add_patch(circle)
            # print(i[0])
    plt.show()


def onclick(self, event):
    if self.mode == 'Manual Mode':
        self.startTime = time.time()
        # print(self.startTime)
    if self.mode == 'Auto Mode':
        self.startTime = time.time()


def onrelease(self, event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    if self.mode == 'Manual Mode':
        if (time.time() - self.startTime < 0.5):
            self.image_click_count += 1
            if self.image_click_count % 2 == 1:
                self.x_center = ix
                self.y_center = iy
                self.xT.set(ix)
                self.yT.set(iy)
                self.x_center = self.xT.get()
                self.y_center = self.yT.get()
            if self.image_click_count % 2 == 0:
                if self.prevCirc is not None:
                    self.prevCirc.set_visible(False)
                self.radius = math.sqrt((float(self.x_center) - ix) ** 2 + (float(self.y_center) - iy) ** 2)
                self.radiusT.set(self.radius)
                self.circle = patches.Circle((float(self.x_center), float(self.y_center)), float(self.radius),
                                             linewidth=1,
                                             edgecolor='r',
                                             facecolor='none')
                self.prevCirc = self.circle

            self.ax.add_patch(self.circle)
            plt.show()
    if self.mode == 'Auto Mode':
        if (time.time() - self.startTime < 0.5):

            minDist = 100000
            for i in self.auto_circles:
                distT = math.sqrt((ix - i[0]) ** 2 + (iy - i[1]) ** 2)
                if distT < minDist:
                    minDist = distT
                    self.circle = patches.Circle((float(i[0]), float(i[1])), float(i[2]),
                                                 linewidth=1,
                                                 edgecolor='r',
                                                 facecolor='none')
                    self.x_center = i[0]
                    self.y_center = i[1]
                    self.radius = i[2]

            if self.prevCirc is not None:
                self.prevCirc.set_visible(False)
            self.xT.set(self.x_center)
            self.yT.set(self.y_center)
            self.radiusT.set(self.radius)
            # print('x center: ' + str(self.x_center))
            self.prevCirc = self.circle
            self.ax.add_patch(self.circle)
            plt.show()


def fileselect_button(self):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    self.filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    # print(self.filename)
    self.image = cv2.imread(self.filename, -1)
    self.prevCirc = None

    self.imgColor = cv2.imread(self.filename, 1)

    self.imgTemp = cv2.imread(self.filename, 0)

    self.image_click_count = 0
    self.fig = plt.figure()
    self.ax = self.fig.add_subplot(111)
    self.ax.imshow(self.image, cmap='gray')

    self.radius = 0
    self.x_center = 0
    self.y_center = 0
    self.xT.set(self.x_center)
    self.yT.set(self.y_center)
    self.radiusT.set(self.radius)

    self.circle = patches.Circle((self.x_center, self.y_center), self.radius, linewidth=1, edgecolor='r',
                                 facecolor='none')
    self.ax.add_patch(self.circle)
    self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
    plt.show()
