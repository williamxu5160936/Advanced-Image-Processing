import matplotlib
import numpy as np
import math
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
import peakutils
import scipy as sp
import scipy.interpolate
from scipy import signal
import cv2

#mode = 1 for cubic spline, 0 for raw data, 2 for bilinear
def interpolate(self, mode, radius):
    if mode == 1:
        z = self.image

        W = np.size(z, 1)
        Y = np.size(z, 0)

        x = np.arange(W)
        y = np.arange(Y)

        xcenter = int(round(float(self.x_center), 0))
        ycenter = int(round(float(self.y_center), 0))
        #r = int(round(float(self.radius), 0))

        interp = sp.interpolate.interp2d(x, y, z, 'cubic')
        vinterp = np.vectorize(interp)

        arc = 8 * np.pi * radius
        ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        val = vinterp(xcenter + radius * np.sin(ang),
                      ycenter + radius * np.cos(ang))

        #returns the angle measures(rad) vs amplitude(greyscale values)
        return ang, val
    if mode == 0:
        z = self.image

        W = np.size(z, 1)
        Y = np.size(z, 0)

        x = np.arange(W)
        y = np.arange(Y)

        xcenter = int(round(float(self.x_center), 0))
        ycenter = int(round(float(self.y_center), 0))
        r = int(round(float(self.radius), 0))

        arc = 8 * np.pi * radius

        circumference_pixels = []
        ang = []
        angle = 0.0
        #while angle < 360:
        while angle < 360:
            xI = int(round(xcenter + radius * math.cos(2.0 * math.pi * angle / 360.0), 0))
            yI = int(round(ycenter + radius * math.sin(2.0 * math.pi * angle / 360.0), 0))
            #angle = angle + (360 / (2 * math.pi * self.radius))
            angle = angle + 0.5

            if xI >= 0 and xI < len(z[0]) and yI >= 0 and yI < len(z):
                circumference_pixels.append(z[yI][xI])
                #ang.append(retAngle(xcenter, ycenter, xI, yI))
                ang.append(angle * np.pi/180)
        ang = np.asarray(ang)
        circumference_pixels = np.asarray(circumference_pixels)
        circumference_pixels = [x for _, x in sorted(zip(ang, circumference_pixels))]
        ang.sort()
        return ang, circumference_pixels
    if mode == 2:
        z = self.image

        W = np.size(z, 1)
        Y = np.size(z, 0)

        x = np.arange(W)
        y = np.arange(Y)

        xcenter = int(round(float(self.x_center), 0))
        ycenter = int(round(float(self.y_center), 0))
        # r = int(round(float(self.radius), 0))

        interp = sp.interpolate.interp2d(x, y, z, 'cubic')
        vinterp = np.vectorize(interp)

        arc = 8 * np.pi * radius
        ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        val = vinterp(xcenter + radius * np.sin(ang),
                      ycenter + radius * np.cos(ang))

        # returns the angle measures(rad) vs amplitude(greyscale values)
        return ang, val

def retAngle(xcenter, ycenter, xI, yI):
    ans = math.degrees(math.atan2(yI - ycenter, xI - xcenter))
    if (ans < 0):
        ans += 360
    return ans * np.pi/180


def drawPlot(self, interp_mode):
    plt.figure()
    r = int(round(self.radius, 0))
    ang, val = interpolate(self, interp_mode, radius = r)
    plt.plot([180/np.pi * i for i in ang], val, label='r={}'.format(r))
    plt.xlabel('degrees from polar axis at r')
    plt.ylabel('pixel greyscale values')
    plt.show()


def drawfft(self, interp_mode):
    r = int(round(self.radius, 0))
    xf, yf, N, func = get_fft(self, interp_mode, radius=r)

    index, properties = signal.find_peaks(func,
                                          height=None,
                                          threshold=None,
                                          distance=5,
                                          prominence=10,
                                          width=None,
                                          wlen=None,
                                          rel_height=None,
                                          plateau_size=None)
    maxs = signal.argrelmax(func)[0]
    mins = signal.argrelmin(func)[0]
    mins = np.insert(mins, 0, 0)

    min = []

    for x in index:
        i = 0
        while i < len(maxs):
            if x == maxs[i]:
                min.append(mins[i])
                if (i + 1 < len(mins)):
                    min.append(mins[i + 1])
                else:
                    min.append(0)
            i += 1
    # i = 0
    # while i < len(index):
    # energy = math.sqrt(sum(abs(func[min[i * 2]:min[i * 2 + 1]]) ** 2))
    # print(energy)
    # i += 1

    fig, ax = plt.subplots()
    ax.plot(xf, func)

    ax.plot(xf[index], func[index], marker="o", ls="", ms=3)

    ax.plot(xf[min], func[min], marker="o", ls="", ms=3)
    peaks, ind = find_peaks(self, interp_mode = 1)
    ax.set(xlim=(0, 1.25*np.max(peaks)), ylim=(0, np.max(func) * 1.25))

    main_x, ind_x = find_main_peak(self)
    main_per = 1/main_x * 180/np.pi
    self.find_main_peak()
    plt.xlabel('frequency(1/rad)')
    plt.ylabel('amplitude')
    plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
              + str(round(main_per, 2)) + ' degrees')

    
    other_x = find_other_peak(self)


    self.find_main_peak()
    plt.xlabel('frequency(1/rad)')
    plt.ylabel('amplitude')
    if (len(other_x) != 0):
        plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
                  + str(round(main_per, 2)) + ' degrees\nOther Peaks: ' + str(other_x))
    else:
        plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
                  + str(round(main_per, 2)) + ' degrees\nOther Peaks: ' + str(round(other_x[0], 2)) + ' rad^-1')

    plt.show()
    plt.show()

def find_other_peak(self):
    xf, yf, N, func = get_fft(self)  
    index, properties = signal.find_peaks(func,
                                          height=max(func)/2,
                                          threshold=None,
                                          distance=None,
                                          prominence=10,
                                          width=None,
                                          wlen=None,
                                          rel_height=None,
                                          plateau_size=None)  
    mainpeak = find_main_peak(self)
    x = xf[index].tolist()
    i = 0
    while(i < len(x)):
        if(abs(x[i] - mainpeak) < 1):
            del x[i]
        i += 1
    #amp = np.interp(x, xf, func)
    return x

def find_amplitude(self, ri):
    en = []
    # indexes of the energies
    en_index = []
    # Number of samplepoints

    N = 720
    # sample spacing: 0.5 degree
    T = np.pi / 360
    z = self.image

    W = np.size(z, 1)
    Y = np.size(z, 0)

    xi = np.arange(W)
    yi = np.arange(Y)

    xcenter = int(round(float(self.x_center), 0))
    ycenter = int(round(float(self.y_center), 0))
    r = int(round(float(self.radius), 0))

    interp = sp.interpolate.interp2d(xi, yi, z, 'cubic')
    vinterp = np.vectorize(interp)

    ang = np.linspace(0, N * T, N, endpoint=False)

    N = ang.size
    y = vinterp(xcenter + ri * np.sin(ang),
                ycenter + ri * np.cos(ang))

    # y = scipy.signal.detrend(y)
    yf = scipy.fftpack.fft(y - np.mean(y))

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    # xf = 1 / xf * 180.0 / np.pi
    func = 2.0 / N * np.abs(yf[:N // 2])

    # index = peakutils.indexes(func)
    index, properties = signal.find_peaks(func,
                                          height=None,
                                          threshold=None,
                                          distance=5,
                                          prominence=10,
                                          width=None,
                                          wlen=None,
                                          rel_height=None,
                                          plateau_size=None)
    peaks_x = peakutils.interpolate(xf, func, ind=index)
    #print(peaks_x)

    i = 0
    while (i < len(peaks_x)):
        en.append(np.interp(peaks_x[i], xf, func))
        i += 1
    return en, xf[index]


def find_main_peak(self):
    r = int(round(float(self.radius), 0))
    en, ind = find_amplitude(self, r)
    p = max_pos(self, en)
    return ind[p], p


def max_pos(self, list):
    max = 0
    index = 0
    for i in range(len(list)):
        if list[i] > max:
            max = list[i]
            index = i
    return index


def get_fft(self, interp_mode, radius):
    # # # Number of samplepoints
    # N = 720
    # # sample spacing: 0.5 degree
    # T = np.pi / 360
    # z = self.image
    #
    # W = np.size(z, 1)
    # Y = np.size(z, 0)
    #
    # xi = np.arange(W)
    # yi = np.arange(Y)
    #
    # xcenter = int(round(float(self.x_center), 0))
    # ycenter = int(round(float(self.y_center), 0))
    # r = int(round(float(self.radius), 0))
    #
    # interp = sp.interpolate.interp2d(xi, yi, z, 'cubic')
    # vinterp = np.vectorize(interp)
    #
    #
    # ang = np.linspace(0, N * T, N, endpoint=False)

    # arc = 8 * np.pi * r
    #         ang = np.linspace(0, 2 * np.pi, int(round(arc * 2,0)), endpoint=False)
    #
    # N = ang.size
    # y = vinterp(xcenter + r * np.sin(ang),
    #             ycenter + r * np.cos(ang))

    #y = scipy.signal.detrend(y)
    ang, y = interpolate(self, interp_mode, radius)
    N = ang.size
    #not evenly spaced for the raw data!
    T = np.pi/360
    yf = scipy.fftpack.fft(y - np.mean(y))

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    func = 2.0 / N * np.abs(yf[:N // 2])
    return xf, yf, N, func


def find_peaks(self, interp_mode):
    # # Number of samplepoints
    # N = 720
    # # sample spacing: 0.5 degree
    # T = np.pi / 360
    # z = self.image
    #
    # W = np.size(z, 1)
    # Y = np.size(z, 0)
    #
    # xi = np.arange(W)
    # yi = np.arange(Y)
    #
    # xcenter = int(round(float(self.x_center), 0))
    # ycenter = int(round(float(self.y_center), 0))
    # r = int(round(float(self.radius), 0))
    #
    # interp = sp.interpolate.interp2d(xi, yi, z, 'cubic')
    # vinterp = np.vectorize(interp)
    #
    # arc = 2 * np.pi * r
    # ang = np.linspace(0, N * T, N, endpoint=False)
    #
    # N = ang.size
    # y = vinterp(xcenter + r * np.sin(ang),
    #             ycenter + r * np.cos(ang))
    #
    # # y = scipy.signal.detrend(y)
    # yf = scipy.fftpack.fft(y - np.mean(y))
    #
    # xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    # func = 2.0 / N * np.abs(yf[:N // 2])
    r = int(round(float(self.radius), 0))
    xf, yf, N, func = get_fft(self, interp_mode, radius=r)
    index, properties = signal.find_peaks(func,
                                          height=None,
                                          threshold=None,
                                          distance=5,
                                          prominence=10,
                                          width=None,
                                          wlen=None,
                                          rel_height=None,
                                          plateau_size=None)
    return xf[index], index


def compute_mtf(self, interp_mode):
    r = int(round(float(self.radius), 0))
    mtf_final = []
    bounds = np.linspace(0, r, r)
    ang, val = interpolate(self, interp_mode, radius=r)
    N = len(ang)

    mtf = np.absolute(scipy.fftpack.fft(val - np.mean(val)))
    func = 2.0 / N * np.abs(mtf[:N // 2])
    max = np.amax(func)
    # print(ind)
    qf, main_index = find_main_peak(self)

    for ri in bounds:

        # N = 720
        # # sample spacing: 0.5 degree
        # T = np.pi / 360
        # z = self.image
        #
        # W = np.size(z, 1)
        # Y = np.size(z, 0)
        #
        # xi = np.arange(W)
        # yi = np.arange(Y)
        #
        # xcenter = int(round(float(self.x_center), 0))
        # ycenter = int(round(float(self.y_center), 0))
        #
        # interp = sp.interpolate.interp2d(xi, yi, z, 'cubic')
        # vinterp = np.vectorize(interp)
        #
        # ang = np.linspace(0, N * T, N, endpoint=False)
        #
        # N = ang.size
        # y = vinterp(xcenter + ri * np.sin(ang),
        #
        #             ycenter + ri * np.cos(ang))
        #
        # mtf = np.absolute(scipy.fftpack.fft(y - np.mean(y)))
        # func = 2.0 / N * np.abs(mtf[:N // 2])
        xf, yf, N, func = get_fft(self, interp_mode, radius=ri)
        peaks, indexes = find_peaks(self, interp_mode)
        ap = 0.0
        for peak in range(len(peaks)):
            if abs(peaks[peak] - qf) < 0.01:
                ap = func[indexes[peak]]
        mtf_final.append(ap)

    xr = np.linspace(0, r, r)
    db_3_interp = np.interp(0.5 * max, mtf_final, xr)
    return qf * 1/xr, mtf_final, qf * 1/db_3_interp, qf*1/self.radius, max


def plot_mtf(self, interp_mode):
    spatial, contrast, spatial_db3, start_spatial, max = compute_mtf(self, interp_mode)
    fig, ax = plt.subplots()
    ax.plot(spatial, contrast)
    ax.plot(spatial_db3, max*.5, marker="o", ls="", ms=3)
    ax.plot(start_spatial, max, marker="o", ls="", ms=3)
    ax.set(xlim=(0, 1.0), ylim=(0, max * 1.25))
    plt.xlabel("spatial frequency(1/pixel)")
    plt.ylabel("amplitude of central peak")
    plt.title('3db point at spatial frequency ' + str(round(spatial_db3, 4)) + ' cycles/pixel ')
    plt.show()

class selection():
    def __init__(self, x, y, r, filename):
        self.image = cv2.imread(filename, -1)
        self.x_center = x
        self.y_center = y
        self.radius = r


if __name__ == '__main__':
    keeprunning = True
    while keeprunning:
        filename = input('Enter File Name: ')
        pattern_center_x = float(input('enter pattern center X coordinate: '))
        pattern_center_y = float(input('enter pattern center Y coordinate: '))
        pattern_radius = float(input('enter radius of selection: '))
        user_select = selection(float(pattern_center_x), float(pattern_center_y), float(pattern_radius), filename)
        task_perform = int(input('enter 1 to find pattern frequency, 2 to find spatial frequency of 3dB point: '))
        if task_perform == 2:
            spatial, contrast, spatial_db3, start_spatial, max = compute_mtf(self=user_select)
            print('3dB point at spatial frequency ' + str(spatial_db3) + ' pixel^-1')
        elif task_perform == 1:
            main_x, ind_x = find_main_peak(self=user_select)
            print('spoke pattern frequency at ' + str(round(main_x, 2)) + ' radian^-1')
        continue_option = float(input('Enter 1 to continue, 0 to quit'))
        if continue_option == 0:
            keeprunning = False


