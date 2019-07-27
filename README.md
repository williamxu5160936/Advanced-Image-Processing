# Advanced-Image-Processing

Advanced Image Processing is a software tool that can be used to analyze image quality, and test imaging systems. The tool is designed 
to analyze image quality based on wheel chair patterns found within the image. The wheel chair pattern provides a spectrum of spatial 
frequencies, so we can analyze the contrast versus as a function of spatial frequency. This is advantageous to Air Force patterns as
Air Force patterns only provide one spatial frequency, so many more patterns would have to be tested in order to compare contrast with 
spatial frequency.

The user is allowed to make a circular selection either by manually clicking(first click = center, second click defines radius), or 
using the automatic mode to allow the computer to identify the patterns. 

Advanced Image Processing allows the user to analyze the frequency components around the circular selection using a FFT, 
and the prominence of the spoke pattern frequency is an indicator of good image quality. Finally, the FFT algorithm can be repeated
for smaller radii, and the amplitude of the spoke pattern peak at each of these radii is plotted against the spatial frequency,
in order to generate a Contrast Transfer Function(CTF) plot.
