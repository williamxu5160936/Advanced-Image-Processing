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

#mode = 1 for cubic spline, 0 for raw data
def interpolate(self, mode):
    if mode == 1:
        z = self.image

        W = np.size(z, 1)
        Y = np.size(z, 0)

        x = np.arange(W)
        y = np.arange(Y)

        xcenter = int(round(float(self.x_center), 0))
        ycenter = int(round(float(self.y_center), 0))
        r = int(round(float(self.radius), 0))

        interp = sp.interpolate.interp2d(x, y, z, 'cubic')
        vinterp = np.vectorize(interp)

        arc = 8 * np.pi * r
        ang = np.linspace(0, 2 * np.pi, int(round(arc * 2,0)), endpoint=False)
        val = vinterp(xcenter + r * np.sin(ang),
                      ycenter + r * np.cos(ang))

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

        plt.figure()

        arc = 8 * np.pi * r

        ang = np.linspace(0, 2 * np.pi, arc * 8, endpoint=False)
        xval = xcenter + r * np.sin(ang)
        yval = ycenter + r * np.cos(ang)

        xvals = []
        yvals = []

        i = 0
        while (i < len(xval)):
            xvals.append(int(round(xval[i])))
            yvals.append(int(round(yval[i])))
            i += 1

        i = 0
        val = []
        while (i < len(xvals)):
            val.append(z[yvals[i]][xvals[i]])
            i += 1
        return ang, val

def drawPlot(self):
    plt.figure()
    ang, val = interpolate(self, 0)
    r = int(round(self.radius, 0))
    plt.plot(ang * 180 / np.pi, val, label='r={}'.format(r))
    plt.xlabel('degrees from polar axis at r')
    plt.ylabel('pixel greyscale values')
    plt.show()


def drawfft(self):
    xf, yf, N, func = get_fft(self)

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
    peaks, ind = find_peaks(self)
    ax.set(xlim=(0, 1.25*np.max(peaks)), ylim=(0, np.max(func) * 1.25))

    main_x, ind_x = find_main_peak(self)
    main_per = 1/main_x * 180/np.pi
    self.find_main_peak()
    plt.xlabel('frequency(1/rad)')
    plt.ylabel('amplitude')
    plt.title('pattern frequency at ' + str(round(main_x, 2)) + ' rad^-1, periodicity at '
              + str(round(main_per, 2)) + ' degrees')
    plt.show()


def find_energy(self, ri):
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
    while i < len(peaks_x):
        half_max = np.interp(peaks_x[i], xf, func) / 2.0
        s = sp.interpolate.UnivariateSpline(xf, func - half_max)
        roots = s.roots()
        closeleft = 0
        closeright = len(xf)
        e = 0
        while e < len(roots) and roots[e] < peaks_x[i]:
            e += 1
        if e == len(roots):
            e = len(roots) - 1
        width = abs(roots[e] - roots[e - 1])
        if width > 0.5:
            width = 0.5
        #print('width": ' + str(width))
        ys = []
        start = peaks_x[i] - width

        a = 0
        num = 5.0
        while a <= num * 2:
            ys.append(np.interp(start, xf, func))
            start += width / num
            a += 1
        energy = 0
        for c in ys:
            energy += (c ** 2 * width / num)
        #if energy < 0:
            #print(energy)
        #print(math.sqrt(energy))
        en.append(math.sqrt(energy))
        i += 1
    #print(en)
    return en, xf[index]


def find_main_peak(self):
    r = int(round(float(self.radius), 0))
    en, ind = find_energy(self, r)
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


def get_fft(self):
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

    arc = 2 * np.pi * r
    ang = np.linspace(0, N * T, N, endpoint=False)

    N = ang.size
    y = vinterp(xcenter + r * np.sin(ang),
                ycenter + r * np.cos(ang))

    # y = scipy.signal.detrend(y)
    yf = scipy.fftpack.fft(y - np.mean(y))

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    func = 2.0 / N * np.abs(yf[:N // 2])
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


def compute_mtf(self):
    r = int(round(float(self.radius), 0))
    mtf_final = []
    bounds = np.linspace(0, r, r)
    ang, val = interpolate(self, 1)
    N = len(ang)

    mtf = np.absolute(scipy.fftpack.fft(val - np.mean(val)))
    func = 2.0 / N * np.abs(mtf[:N // 2])
    max = np.amax(func)
    # print(ind)
    qf, main_index = find_main_peak(self)

    for ri in bounds:
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

        ang = np.linspace(0, N * T, N, endpoint=False)

        N = ang.size
        y = vinterp(xcenter + ri * np.sin(ang),

                    ycenter + ri * np.cos(ang))

        mtf = np.absolute(scipy.fftpack.fft(y - np.mean(y)))
        func = 2.0 / N * np.abs(mtf[:N // 2])
        #n, positions = find_energy(self, ri)
        # print('main peak')

        peaks, indexes = find_peaks(self)
        #print('qf' + str(qf))
        ap = 0.0
        for peak in range(len(peaks)):
            if abs(peaks[peak] - qf) < 0.01:
                ap = func[indexes[peak]]
        mtf_final.append(ap)
        # print((np.max(y) - np.min(y)) / np.amax(mtf))

        # ---------------------energy

        # print(positions)
        # if len(n) > 0:
        #     q = 0.0
        #     for i in range(len(positions)):
        #         if math.fabs(positions[i] - qf) < 0.01:
        #             q = n[i]
        #
        #     yen.append(q / np.sum(n))
        # else:
        #     yen.append(0.0)

    # mtf_final_smooth = mtf_final_smooth[1024:1151] / np.amax(mtf_final_smooth[1024:1151])
    xr = np.linspace(0, r, r)

    #fig, ax = plt.subplots()
    #start = np.interp(max, mtf_final, xr)
    db_3_interp = np.interp(0.5 * max, mtf_final, xr)
    #ax.plot(1/(2.0 * np.pi * start), max, marker="o", ls="", ms=3)
    #ax.plot(1/(2.0 * np.pi * db_3_interp), 0.5 * max, marker="o", ls="", ms=3)
    # ax.plot(1/(2.0 * np.pi * xr), mtf_final)
    # ax.set(xlim=(0, 0.05), ylim=(0, max*1.25))
    # plt.xlabel("spatial frequency(AU)")
    # plt.ylabel("amplitude of central peak")
    # plt.title('3dB point at spatial frequency' + str(round(1/(2.0 * np.pi * db_3_interp), 4)))
    # plt.show()

    return qf * 1/xr, mtf_final, qf * 1/db_3_interp, qf*1/self.radius, max

def plot_mtf(self):
    spatial, contrast, spatial_db3, start_spatial, max = compute_mtf(self)
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




