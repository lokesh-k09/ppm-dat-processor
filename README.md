# PPM DAT Processor

A web application for processing `.dat` files containing time-domain or spectral data for PPM/FWHM analysis. This tool extracts, analyzes, and visualizes spectral properties, exporting both plots and Excel reports for quality assurance (QA) and scientific review.

---

## Features

- **Drag-and-drop interface** for uploading one or multiple `.dat` files
- **Automatic extraction** of acquisition parameters and signal data
- **FFT-based spectrum analysis**: calculates FWHM (Full Width at Half Maximum) in Hz and ppm
- **Radar/polar plot** visualizing homogeneity vs. gradient angle
- **Excel reports**: all summary results and raw FFT data in a single multi-sheet file
- **Spectrum plots**: generates publication-quality JPG images for each dataset
- **Batch processing**: handles multiple files efficiently
- **Zipped results** for easy download

---

## Usage

1. **Run or deploy the app**  
   - [Recommended] **Streamlit Cloud**: Deploy directly from this repository  
   - **Local:**  
     ```
     git clone https://github.com/lokesh-k09/ppm-dat-processor.git
     cd ppm-dat-processor
     pip install -r requirements.txt
     streamlit run app.py
     ```

2. **Upload Files**  
   - Use the Streamlit web interface to upload your `.dat` files (single or batch).

3. **Review Results**  
   - Download the zipped results, which include:
     - Individual spectrum plots (JPG)
     - Polar plot (PNG)
     - An Excel report (multi-sheet) with summary and raw spectra

---

## Requirements

- Python 3.8+
- Streamlit
- Numpy
- Pandas
- Matplotlib
- twixtools
- openpyxl
- (see `requirements.txt`)

---

## How it works

1. Reads and parses `.dat` files using `twixtools`
2. Extracts relevant acquisition and signal information
3. Performs FFT, combines channels, and normalizes spectra
4. Calculates FWHM and generates spectrum plots
5. Aggregates all results into an Excel file and images
6. Packages everything into a downloadable ZIP

---



- Developed by [Lokesh K.](https://github.com/lokesh-k09)
- Uses [`twixtools`](https://github.com/mrirecon/twixtools) for `.dat` file parsing

---
