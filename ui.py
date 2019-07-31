import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import tkinter as tk
import analysis as an
import user_controls as uc


class UIProject(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("300x375")
        self.resizable(0, 0)

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = StartPage(container, self)
        self.frames[StartPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)


    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        bottomFrame = Frame(self)
        bottomFrame.pack(side=BOTTOM, fill=BOTH)

        label = tk.Label(self, text="   Advanced Image Processing")
        label.pack(pady=10, padx=10)
        self.mode = 'Manual Mode'

        select_button = tk.Button(self, text="Select File",
                                  command=self.fileselect_button)
        select_button.pack(fill=BOTH)

        self.manual_button = tk.Button(self, text="Manual Mode", command=self.manual_mode)
        self.manual_button.pack(fill=BOTH)
        self.auto_button = tk.Button(self, text="Auto Mode", command=self.auto_mode)
        self.auto_button.pack(fill=BOTH)
        self.setval_button = tk.Button(self, text="Press to Set Values", command=self.ok_pressed)
        self.setval_button.pack(fill=BOTH)
        self.pixel_plot = tk.Button(self, text="Plot Data", command=self.draw_pixel_plot)
        self.pixel_plot.pack(fill=BOTH)

        self.pixelplot0_button = tk.Button(self, text="Raw Data", command=self.drawPlotNearest)
        self.pixelplot0_button.pack(side=LEFT, fill=BOTH)

        self.pixelplot1_button = tk.Button(self, text="Interpolate Linear", command=self.drawPlotLinear)
        self.pixelplot1_button.pack(side=LEFT, fill=BOTH)

        self.pixelplot3_button = tk.Button(self, text="Interpolate Cubic", command=self.drawPlotCubic)
        self.pixelplot3_button.pack(side=LEFT, fill=BOTH)


        self.fft_button = tk.Button(bottomFrame, text="Generate FFT Plot(frq v amp)", command=self.drawfft)
        self.fft_button.pack(fill=BOTH)


        self.ctf_button = tk.Button(bottomFrame, text="Generate CTF Plot", command=self.plot_ctf)
        self.ctf_button.pack(fill=BOTH)

        self.radiusT = StringVar()
        radius_label = tk.Label(bottomFrame, text="Radius:").pack()
        self.radius_box = Entry(bottomFrame, textvariable=self.radiusT, width=25, bg="Lightgreen").pack()
        self.radiusT.set("0")
        self.radius = int(self.radiusT.get())

        self.xT = StringVar()
        xCenter_label = tk.Label(bottomFrame, text="X coordinate of pattern center:").pack()
        self.xCenter_box = Entry(bottomFrame, textvariable=self.xT, width=25, bg="Lightgreen").pack()
        self.xT.set("0")
        self.x_center = int(self.xT.get())

        self.yT = StringVar()
        yCenter_label = tk.Label(bottomFrame, text="Y coordinate of pattern center:").pack()
        self.yCenter_box = Entry(bottomFrame, textvariable=self.yT, width=25, bg="Lightgreen").pack()
        self.yT.set("0")
        self.y_center = int(self.yT.get())

        #interpolation mode defaults to cubic spline
        self.interp = 3

    def ok_pressed(self):
        uc.ok_pressed(self)

    def manual_mode(self):
        uc.manual_mode(self)

    def auto_mode(self):
        uc.auto_mode(self)

    def onclick(self, event):
        uc.onclick(self, event)

    def onrelease(self, event):
        uc.onrelease(self, event)

    def fileselect_button(self):
        uc.fileselect_button(self)

    def drawPlotNearest(self):
        self.interp = 0

    def drawPlotLinear(self):
        self.interp = 1

    def drawPlotCubic(self):
        self.interp = 3

    def draw_pixel_plot(self):
        an.drawPlot(self, self.interp)

    def drawfft(self):
        an.drawfft(self)

    def plot_ctf(self):
        an.plot_ctf(self)



app = UIProject()
app.mainloop()
