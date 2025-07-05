import streamlit as st
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ⚠️ Replace this with your working map_twix
from twixtools.map_twix import map_twix

def process_dat_file(dat_file, output_dir):
    twix = map_twix(dat_file)

    dwelltime = twix[0]['hdr']['MeasYaps']['sRXSPEC']['alDwellTime'][0]
    Nfreq = 256
    bw_per_pixel = (1e9 / (dwelltime * Nfreq))
    CF = twix[0]['hdr']['Dicom']['lFrequency']

    image_obj = twix[0]['image']
    imagedata = np.asarray(image_obj)
    if imagedata.ndim != 3:
        st.warning(f"Skipped: {dat_file} → Not 3D data.")
        return None, None

    nx, nc, nr = imagedata.shape
    ppmksp = imagedata.reshape((nx, nc * nr))

    ppmfft = np.abs(np.fft.fftshift(np.fft.ifft(ppmksp, axis=0), axes=0))
    ppmdata = np.sqrt(np.sum(ppmfft ** 2, axis=1))
    ppmdata /= np.max(ppmdata)

    # Frequency axis
    N = len(ppmdata)
    total_bw = bw_per_pixel * N
    xvals = np.linspace(-total_bw/2, total_bw/2, N)

    # FWHM
    peakVal = np.max(ppmdata)
    peakIdx = np.argmax(ppmdata)
    halfVal = peakVal / 2
    delta_cf = round((peakIdx - (N/2)) * bw_per_pixel)
    CF_New = CF + delta_cf

    leftIdx = np.where(ppmdata[:peakIdx] <= halfVal)[0][-1] if np.any(ppmdata[:peakIdx] <= halfVal) else 0
    rightIdx = peakIdx + np.where(ppmdata[peakIdx:] <= halfVal)[0][0] if np.any(ppmdata[peakIdx:] <= halfVal) else N-1

    xLeft = np.interp(halfVal, [ppmdata[leftIdx], ppmdata[leftIdx+1]], [xvals[leftIdx], xvals[leftIdx+1]]) if leftIdx+1 < N else xvals[0]
    xRight = np.interp(halfVal, [ppmdata[rightIdx-1], ppmdata[rightIdx]], [xvals[rightIdx-1], xvals[rightIdx]]) if rightIdx < N else xvals[-1]

    FWHM_Hz = xRight - xLeft
    FWHM_ppm = FWHM_Hz / CF_New * 1e6

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(xvals, ppmdata)
    plt.plot([xLeft, xRight], [halfVal, halfVal], 'r-', linewidth=2)
    plt.title(f'FFT: {os.path.basename(dat_file)}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Amplitude')

    plot_path = os.path.join(output_dir, f"{os.path.basename(dat_file)}.png")
    plt.savefig(plot_path)
    plt.close()

    return (CF_New, FWHM_Hz, FWHM_ppm), ppmdata


# ---------------------------------
st.title("Siemens .DAT Processor")

uploaded_files = st.file_uploader(
    "Drop or browse .dat files here",
    type=["dat"],
    accept_multiple_files=True
)

if uploaded_files:
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ppm_summary = []
    fft_data_all = []
    labels = []

    for f in uploaded_files:
        st.write(f"Processing: {f.name}")
        dat_path = os.path.join(output_dir, f.name)
        with open(dat_path, "wb") as out:
            out.write(f.read())

        summary, fftdata = process_dat_file(dat_path, output_dir)
        if summary:
            ppm_summary.append(summary)
            fft_data_all.append(fftdata)
            labels.append(f.name)

    if ppm_summary:
        df = pd.DataFrame(ppm_summary, columns=["CF", "FWHM Hz", "FWHM ppm"])
        df["Label"] = labels
        df.to_excel(os.path.join(output_dir, "PPM_Summary.xlsx"), index=False)

        fft_df = pd.DataFrame(np.array(fft_data_all).T, columns=labels)
        fft_df.to_excel(os.path.join(output_dir, "FFT_Data.xlsx"), index=False)

        st.success(f"✅ All done! Files saved in: {output_dir}")
    else:
        st.warning("No valid files processed.")

