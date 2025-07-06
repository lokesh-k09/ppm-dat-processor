import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from io import BytesIO
import tempfile

from twixtools.map_twix import map_twix


def process_dat_file(dat_file, plotsdir):
    twix = map_twix(dat_file)

    dwelltime = twix[0]['hdr']['MeasYaps']['sRXSPEC']['alDwellTime'][0]
    Nfreq = 256
    bw_per_pixel = (1e9 / (dwelltime * Nfreq))
    CF = twix[0]['hdr']['Dicom']['lFrequency']

    imagedata = np.squeeze(np.asarray(twix[0]['image'][:]))
    if imagedata.ndim < 2:
        st.warning(f"❌ Not enough usable image data → {dat_file}")
        return None, None, None

    if imagedata.shape[-2] == 2:
        coils = int(np.prod(imagedata.shape[:-2]))
        i_q, nfreq = imagedata.shape[-2:]
        imagedata = imagedata.reshape((coils, i_q, nfreq))
        ppmksp = imagedata[:, 0, :] + 1j * imagedata[:, 1, :]
    else:
        st.warning(f"❌ Unexpected shape {imagedata.shape}")
        return None, None, None

    ppmfft = np.abs(np.fft.fftshift(np.fft.ifft(ppmksp, axis=1), axes=1))
    ppmdata = np.sqrt(np.sum(ppmfft ** 2, axis=0))
    ppmdata /= np.max(ppmdata)

    N = len(ppmdata)
    total_bw = bw_per_pixel * N
    xvals = np.linspace(-total_bw / 2, total_bw / 2, N)

    peakVal = np.max(ppmdata)
    peakIdx = np.argmax(ppmdata)
    halfVal = peakVal / 2
    delta_cf = round((peakIdx - (N / 2)) * bw_per_pixel)
    CF_New = CF + delta_cf

    leftIdx = np.where(ppmdata[:peakIdx] <= halfVal)[0][-1] if np.any(ppmdata[:peakIdx] <= halfVal) else 0
    rightIdxRel = np.where(ppmdata[peakIdx:] <= halfVal)[0][0] if np.any(ppmdata[peakIdx:] <= halfVal) else N - 1
    rightIdx = peakIdx + rightIdxRel

    xLeft = np.interp(halfVal, [ppmdata[leftIdx], ppmdata[leftIdx + 1]], [xvals[leftIdx], xvals[leftIdx + 1]]) \
        if leftIdx + 1 < N else xvals[0]
    xRight = np.interp(halfVal, [ppmdata[rightIdx - 1], ppmdata[rightIdx]], [xvals[rightIdx - 1], xvals[rightIdx]]) \
        if rightIdx < N else xvals[-1]

    FWHM_Hz = xRight - xLeft
    FWHM_ppm = FWHM_Hz / CF_New * 1e6

    base = os.path.basename(dat_file)
    GA_label = base.split("_")[-1].replace(".dat", "")
    title = f'FID GA = {GA_label}'

    plt.figure(figsize=(12, 6))
    plt.plot(xvals, ppmdata, 'b', label='Spectrum')
    plt.plot([xLeft, xRight], [halfVal, halfVal], 'k-', lw=5, label='FWHM')

    plt.xlim(-500, 500)
    plt.ylim(0, 1.05)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Normalized Amplitude')
    plt.title(title, fontsize=12, fontweight='bold')

    txtX = 140
    plt.text(txtX, 0.85, f'CF = {CF_New:.0f} Hz', weight='bold')
    plt.text(txtX, 0.75, f'FWHM = {FWHM_Hz:.2f} Hz', weight='bold')
    plt.text(txtX, 0.65, f'FWHM = {FWHM_ppm:.2f} ppm', weight='bold')
    plt.text(txtX, 0.35, 'GO$_x$ = uT/m or DAQ')
    plt.text(txtX, 0.25, 'GO$_y$ = uT/m or DAQ')
    plt.text(txtX, 0.15, 'GO$_z$ = uT/m or DAQ')

    plt.grid(True)
    plt.savefig(os.path.join(plotsdir, f"{base}.jpg"), bbox_inches='tight', dpi=150)
    plt.close()

    return [CF_New, FWHM_Hz, FWHM_ppm], ppmdata, GA_label


st.title(".DAT Processor")

uploaded_files = st.file_uploader("Drop or browse .dat files here", type=['dat'], accept_multiple_files=True)

if uploaded_files:
    datdir = tempfile.mkdtemp()
    qadir = os.path.join(datdir, 'QA')
    plotsdir = os.path.join(qadir, 'PLOTS_PPM')
    exceldir = os.path.join(qadir, 'Excels')

    for path in [qadir, plotsdir, exceldir]:
        os.makedirs(path, exist_ok=True)

    ppm_summary = []
    fft_data_all = []
    GA_labels = []
    nx = 0

    for f in uploaded_files:
        local_file = os.path.join(datdir, f.name)
        with open(local_file, 'wb') as out:
            out.write(f.read())

        st.write(f"📂 Processing: {f.name}")
        summary, fftdata, ga_label = process_dat_file(local_file, plotsdir)
        if summary:
            ppm_summary.append(summary)
            fft_data_all.append(fftdata)
            GA_labels.append(ga_label)
            nx = len(fftdata)

    if ppm_summary:
        summary_df = pd.DataFrame(ppm_summary, columns=['CF', 'FWHM Hz', 'FWHM PPM'])
        summary_df.insert(0, 'GA_Label', GA_labels)
        summary_df.to_excel(os.path.join(exceldir, 'PPM_Summary.xlsx'), index=False)

        fft_df = pd.DataFrame(np.array(fft_data_all).T, columns=GA_labels)
        fft_df.insert(0, 'SampleIdx', np.arange(1, nx + 1))
        fft_df.to_excel(os.path.join(exceldir, 'PPM_Info_full.xlsx'), index=False)

        angle_deg = np.array([int(label) for label in GA_labels])
        idx = np.argsort(angle_deg)
        theta = np.deg2rad(np.append(angle_deg[idx], angle_deg[idx][0]))
        r_ppm = np.append(summary_df['FWHM PPM'].values[idx], summary_df['FWHM PPM'].values[idx][0])
        mean_circle = np.mean(r_ppm) * np.ones_like(theta)

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(theta, r_ppm, 'k-', lw=1)
        ax.plot(theta, mean_circle, 'r-', lw=1)
        ax.plot(theta, r_ppm, 'g*', markersize=6)
        ax.set_title('HOMOGENEITY v. GA', weight='bold')
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_rmax(10)
        plt.savefig(os.path.join(plotsdir, 'Homogeneity_vs_GA_POLAR_PPM.png'), bbox_inches='tight', dpi=150)
        plt.close()

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for folder, _, files in os.walk(qadir):
                for file in files:
                    file_path = os.path.join(folder, file)
                    zipf.write(file_path, os.path.relpath(file_path, qadir))
        zip_buffer.seek(0)

        st.download_button(
            label="📥 Download QA Results ZIP",
            data=zip_buffer,
            file_name="QA_Results.zip",
            mime="application/zip"
        )

        st.success("✅ Done! Click the button above to download your results.")
    else:
        st.warning("⚠️ No valid files processed.")
