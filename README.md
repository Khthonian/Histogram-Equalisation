# Histogram Equalisation
An implementation of histogram equalisation, using C++ and OpenCL, to showcase the usage of parallel programming.

## Description
- The code is designed to handle 8-bit and 16-bit imagery, of both greyscale and RGB varieties.
- The contents of this code contain adaptations and improvements of Tutorial 2 and Tutorial 3, for the base code and the kernel functions. This can be found at https://github.com/wing8/OpenCL-Tutorials
- There is also a kernel function that has been adapted from https://github.com/spoolean/HistogramEqualisation
- The images that the code was tested on include .ppm and .pgm images. These images can be found in the relevant directories.
- The intHistogram2 and cumHistogramHS2 kernels require extra arguments to be passed and these can be uncommented and commented as necessary, and are labelled accordingly.
- The intensity histogram implementations feature a serial implementation and a parallel reduction implementation.
- The cumulative histogram implementations feature a simple implementation, two variations of the Hillis-Steele pattern, and a single implementation of the Blelloch pattern.
- The user is able to give their own desired bin count, which can affect the output of the image and the histograms produced.
- Performance metrics and the histograms are displayed to the user via the console.
- Each step of the model will be indicated as follows: "STEP X - XXXXX"

## Issues
- The 16-bit functionality is only produces a suitable image using a combination of the intHistogram and cumHistogram kernel functions.
- The cumHistogramHS kernel function calculates a histogram but does not produce a suitable image.
