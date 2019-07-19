import matplotlib
import numpy as np
import math
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
import peakutils
import scipy as sp
import scipy.interpolate
from scipy import signal

def drawPlot(self):
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

    plt.figure()

    arc = 8 * np.pi * r
    ang = np.linspace(0, 2 * np.pi, arc * 2, endpoint=False)
    val = vinterp(xcenter + r * np.sin(ang),
                  ycenter + r * np.cos(ang))
    plt.plot(ang * 180 / np.pi, val, label='r={}'.format(r))
    plt.xlabel('degrees from polar axis at r')
    plt.ylabel('pixel values')
    plt.show()


def drawfft(self):
    xf, yf, N, func = get_fft(self)

    index, properties = signal.find_peaks(func,
                                          height=None,
                                          threshold=None,
                                          distance=None,
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
                min.append(mins[i + 1])
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

    self.find_main_peak()
    plt.xlabel('frequency(Hz)')
    plt.ylabel('amplitude')
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
    print(peaks_x)

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
        print('width": ' + str(width))
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
        if energy < 0:
            print(energy)
        print(math.sqrt(energy))
        en.append(math.sqrt(energy))
        i += 1
    print(en)
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


def draw_deg_fft(self):
    # Number of samplepoints
    xf, yf, N, func = get_fft(self)

    xf = 1 / xf * 180.0 / np.pi
    fig, ax = plt.subplots()

    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.xlabel('Degree Measure')
    plt.ylabel('amplitude')
    plt.show()


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
    yen = []
    bounds = np.linspace(0, r, r)
    N = 1440
    # sample spacing: 0.5 degree
    T = np.pi / 720
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
    y = vinterp(xcenter + r * np.sin(ang),
                ycenter + r * np.cos(ang))

    mtf = np.absolute(scipy.fftpack.fft(y - np.mean(y)))
    func = 2.0 / N * np.abs(mtf[:N // 2])
    max = np.amax(func)
    #en_max, ind = find_energy(self, r)
    # print(ind)
    qf, main_index = find_main_peak(self)
    for ri in bounds:
        N = 1440
        # sample spacing: 0.5 degree
        T = np.pi / 720
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

    fig, ax = plt.subplots()
    # #plt.subplot(2, 1, 1)
    #db_interp = sp.interpolate.interp1d(xr, mtf_final, 'cubic')
    db_3_interp = np.interp(0.5 * max, mtf_final, xr)
    #print('db3 interp')
    ax.plot(1/(2.0 * np.pi * db_3_interp), 0.5 * max, marker="o", ls="", ms=3)
    ax.plot(1/(2.0 * np.pi * xr), mtf_final)
    ax.set(xlim=(0, 0.05), ylim=(0, max*1.25))
    #plt.yscale("log")
    #plt.xscale("log")
    plt.xlabel("spatial frequency(AU)")
    plt.ylabel("amplitude of central peak")
    plt.title('3dB point at spatial frequency' + str(round(1/(2.0 * np.pi * db_3_interp), 4)))
    plt.show()

    # fig, ax = plt.subplots()
    # # #plt.subplot(2, 1, 1)
    # # db_interp = sp.interpolate.interp1d(xr, mtf_final, 'cubic')
    # db_3_interp = np.interp(0.5 * max, mtf_final, xr)
    # # print('db3 interp')
    # ax.plot(qf * 1 / db_3_interp, 0.5 * max, marker="o", ls="", ms=3)
    # ax.plot(qf * 1 / xr, mtf_final)
    # # ax.set(xlim=(0, 0.05), ylim=(0, max*1.25))
    # # plt.yscale("log")
    # # plt.xscale("log")
    # plt.xlabel("spatial frequency(AU)")
    # plt.ylabel("amplitude of central peak")
    # plt.title('3db point at spatial frequency' + str(round(qf * 1 / db_3_interp, 4)))
    # plt.show()
    #need to find 3db point


    # plt.subplot(2, 1, 2)

    # ax.plot(xr, yen)
    # plt.xlabel('Radius')
    # plt.ylabel('Energy pctg of Central Peak')
    # plt.show()
