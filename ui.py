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
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page")
        label.pack(pady=10, padx=10)
        self.mode = 'Manual Mode'

        select_button = tk.Button(self, text="Select File",
                                  command=self.fileselect_button)
        select_button.pack()
        self.manual_button = tk.Button(self, text="Manual Mode", command=self.manual_mode)
        self.manual_button.pack()
        self.auto_button = tk.Button(self, text="Auto Mode", command=self.auto_mode)
        self.auto_button.pack()
        self.setval_button = tk.Button(self, text="press ok to set values", command=self.ok_pressed)
        self.setval_button.pack()

        self.pixelplot_button = tk.Button(self, text="pixel plot", command=self.drawPlot)
        self.pixelplot_button.pack()

        self.fft_button = tk.Button(self, text="generate FFT plot(frq v amp)", command=self.drawfft)
        self.fft_button.pack()

        self.fft_degrees_button = tk.Button(self, text="FFT plot(amplitude at angle(deg)", command=self.draw_deg_fft)
        self.fft_degrees_button.pack()

        self.mtf_button = tk.Button(self, text="generate MTF plot", command=self.compute_mtf)
        self.mtf_button.pack()

        self.radiusT = StringVar()
        radius_label = tk.Label(self, text="Radius").pack(pady=30)
        self.radius_box = Entry(self, textvariable=self.radiusT, width=25, bg="Lightgreen").place(x=180, y=272)
        self.radiusT.set("0")
        self.radius = int(self.radiusT.get())

        self.xT = StringVar()
        xCenter_label = tk.Label(self, text="X coordinate of pattern center:").pack(pady=0)
        self.xCenter_box = Entry(self, textvariable=self.xT, width=25, bg="Lightgreen").place(x=180, y=322)
        self.xT.set("0")
        self.x_center = int(self.xT.get())

        self.yT = StringVar()
        yCenter_label = tk.Label(self, text="Y coordinate of pattern center:").pack(pady=30)
        self.yCenter_box = Entry(self, textvariable=self.yT, width=25, bg="Lightgreen").place(x=180, y=372)
        self.yT.set("0")
        self.y_center = int(self.yT.get())

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

    def drawPlot(self):
        an.drawPlot(self)

    def drawfft(self):
        an.drawfft(self)

    def find_energy(self, ri):
        an.find_energy(self, ri)

    def find_main_peak(self):
        an.find_main_peak(self)

    def max_pos(self, list):
        an.max_pos(self, list)

    def get_fft(self):
        an.get_fft(self)

    def draw_deg_fft(self):
        an.draw_deg_fft(self)

    def find_peaks(self):
        an.find_peaks(self)

    def compute_mtf(self):
        an.compute_mtf(self)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One")
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Back to Start Page",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = UIProject()
app.mainloop()
