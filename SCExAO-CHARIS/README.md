# CHARIS Data-Processing Pipeline - Excerpts
Example code from the CHARIS Data-Processing Pipeline, for processing and analyzing SCExAO/CHARIS integral field spectrograph data cubes.   
The IDL pipeline can be found [here](https://github.com/thaynecurrie/charis-dpp); Python pipeline release forthcoming.

## interpolate.py
Performs cubic convolution interpolation.  
Developed to mimic [IDL's proprietary INTERPOLATE function](https://www.l3harrisgeospatial.com/docs/interpolate.html).  
Adapted from Rifman, S.S. and McKinnon, D.M., “Evaluation of Digital Correction Techniques for ERTS Images; Final Report”, Report 20634-6003-TU-00, TRW Systems, Redondo Beach, CA, July 1974.

## aloci_fwdmod_planet.py
Performs forward-modeling to determine the signal loss per channel of a planet at a given position, and outputs throughput correction.  
Translated into Python from IDL code. 

## extract_1d_spectrum.py
Extracts the spectrum of a planet from a PSF-subtracted CHARIS data cube, with or without throughput correction.   
Translated into Python from IDL code. 
