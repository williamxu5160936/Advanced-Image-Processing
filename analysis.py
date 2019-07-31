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

def interpolate(self, mode):
    z = self.image

    W = np.size(z, 1)
    Y = np.size(z, 0)

    x = np.arange(W)
    y = np.arange(Y)
    r = int(round(float(self.radius), 0))
    
    xcenter = int(round(float(self.x_center), 0))
    ycenter = int(round(float(self.y_center), 0))
    
    if mode == 0:
        arc = 8 * np.pi * r

        circumference_pixels = []
        ang = []
        angle = 0.0
        while angle < 360:
            xI = int(round(xcenter + r * math.cos(2.0 * math.pi * angle / 360.0), 0))
            yI = int(round(ycenter + r * math.sin(2.0 * math.pi * angle / 360.0), 0))
            angle = angle + (360 / (2 * math.pi * self.radius))
            if xI >= 0 and xI < len(z[0]) and yI >= 0 and yI < len(z):
                circumference_pixels.append(z[yI][xI])
                ang.append(retAngle(xcenter, ycenter, xI, yI))
        ang = np.asarray(ang)
        circumference_pixels = np.asarray(circumference_pixels)
        circumference_pixels = [x for _,x in sorted(zip(ang,circumference_pixels))]
        ang.sort()
        plt.title("raw data")
        return ang, circumference_pixels
    if mode == 1:
        interp = sp.interpolate.interp2d(x, y, z, 'linear')
        plt.title("linear interpolation")
    if mode == 3:
        interp = sp.interpolate.interp2d(x, y, z, 'cubic')
        plt.title("cubic interpolation")
        
    vinterp = np.vectorize(interp)

    arc = 8 * np.pi * r
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    val = vinterp(xcenter + r * np.sin(ang),
                  ycenter + r * np.cos(ang))
    return ang, val

def retAngle(xcenter, ycenter, xI, yI):
    ans = math.degrees(math.atan2(yI-ycenter, xI-xcenter))
    if(ans < 0):
        ans += 360
    return ans

def drawPlot(self, mode):
    plt.figure()
    ang, val = interpolate(self, mode)
    r = int(round(self.radius, 0))
    plt.plot(ang, val, label='r={}'.format(r))
    plt.xlabel('degrees from polar axis at r')
    plt.ylabel('pixel greyscale values')
    plt.show()


def drawfft(self):
    xf, yf, N, func = get_fft(self, self.radius)

    peaks, index = find_peaks(self)

    fig, ax = plt.subplots()
    ax.plot(xf, func)

    ax.set(xlim=(0, 1.25*np.max(peaks)), ylim=(0, np.max(func) * 1.25))

    main_x = find_main_peak(self)
    other_x = find_other_peak(self)
    main_per = 1/main_x * 180/np.pi
    
    plt.xlabel('frequency(1/rad)')
    plt.ylabel('amplitude')
    if(len(other_x) != 0):
        plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
                  + str(round(main_per, 2)) + ' degrees\nSecondary Peaks: ' + str(round(other_x[0], 2)) + ' rad^-1')
    else:
        plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
                  + str(round(main_per, 2)) + ' degrees')
    plt.show()

    
def find_amplitude(self):
    xf, yf, N, func = get_fft(self, self.radius)
    peaks, index = find_peaks(self)
    amp = []
    
    peaks_x = peakutils.interpolate(xf, func, ind=index)
    i = 0
    while (i < len(peaks_x)):
        amp.append(np.interp(peaks_x[i], xf, func))
        i += 1
    return amp, xf[index]


def find_main_peak(self):
    amp, ind = find_amplitude(self)
    max = 0
    p = 0
    for i in range(len(amp)):
        if amp[i] > max:
            max = amp[i]
            p = i
    return ind[p]

def find_other_peak(self):
    xf, yf, N, func = get_fft(self, self.radius)  
    peaks, index = find_peaks(self)
    mainpeak = find_main_peak(self)
    x = xf[index].tolist()
    i = 0
    while(i < len(x)):
        if(abs(x[i] - mainpeak) < 1):
            del x[i]
        i += 1
    return x


def get_fft(self, r):
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

    interp = sp.interpolate.interp2d(xi, yi, z, 'cubic')
    vinterp = np.vectorize(interp)

    arc = 2 * np.pi * r
    ang = np.linspace(0, N * T, N, endpoint=False)

    N = ang.size
    y = vinterp(xcenter + r * np.sin(ang),
                ycenter + r * np.cos(ang))

    # y = scipy.signal.detrend(y)
    yf = scipy.fftpack.fft(y - np.mean(y))

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    func = 2.0 / N * np.abs(yf[:N // 2])
    return xf, yf, N, func


def find_peaks(self):
    xf, yf, N, func = get_fft(self, self.radius)
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


def compute_ctf(self):
    r = int(round(float(self.radius), 0))
    ctf_final = []
    bounds = np.linspace(0, r, r)
    ang, val = interpolate(self, 3)
    N = len(ang)

    ctf = np.absolute(scipy.fftpack.fft(val - np.mean(val)))
    func = 2.0 / N * np.abs(ctf[:N // 2])
    max = np.amax(func)
    # print(ind)
    qf = find_main_peak(self)

    for ri in bounds:
        xf, yf, N, func = get_fft(self, ri)

        peaks, indexes = find_peaks(self)
        ap = 0.0
        for peak in range(len(peaks)):
            if abs(peaks[peak] - qf) < 0.01:
                ap = func[indexes[peak]]
        ctf_final.append(ap)
    xr = np.linspace(0, r, r)
    db_3_interp = np.interp(0.5 * max, ctf_final, xr)

    return qf * 1/xr, ctf_final, qf * 1/db_3_interp, qf*1/self.radius, max

def plot_ctf(self):
    spatial, contrast, spatial_db3, start_spatial, max = compute_ctf(self)
    fig, ax = plt.subplots()
    ax.plot(spatial, contrast)
    ax.plot(spatial_db3, max*.5, marker="o", ls="", ms=3)
    ax.plot(start_spatial, max, marker="o", ls="", ms=3)
    ax.set(xlim=(0, 1.0), ylim=(0, max * 1.25))
    plt.xlabel("spatial frequency(1/pixel)")
    plt.ylabel("amplitude of central peak")
    plt.title('3db point at spatial frequency ' + str(round(spatial_db3, 4)) + ' cycles/pixel ')
    plt.show()


