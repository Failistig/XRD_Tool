# XRD Analysis Tool

This Python-based analysis tool provides processing, visualization, and evaluation of X-ray diffraction (XRD) data, including **background subtraction**, **peak fitting**, and **Williamson–Hall (W–H) analysis**. 

## Features

- **Background subtraction** using polynomial fitting  
- **Peak identification** via `scipy.signal.find_peaks`  
- **Voigt peak fitting** to extract accurate FWHM and intensity values  
- **Williamson–Hall (W–H) analysis** for microstrain (ε) and crystallite size (L)  
- **Automated plotting** of raw, background-corrected, and fitted data  

## Requirements

Python ≥ 3.9  
Recommended packages:
numpy
scipy
matplotlib
pandas
lmfit
