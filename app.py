# Importing Libraries

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import tempfile
from io import BytesIO
from twixtools.map_twix import map_twix


# This function processed a Siemens MRI .dat file in TWIX format:
# It loaded the scan data and parameters using the twixtools package,
# extracted timing and frequency information, checked the quality of the data,
# and prepared it for further MRI analysis or reporting.
def process_dat_file(dat_file, plotsdir):
    twix = map_twix(dat_file)                                      # Loaded all the scan data and settings from the Siemens file for processing
    dwelltime = float(twix[0]['hdr']['MeasYaps']['sRXSPEC']['alDwellTime'][0])     # Got the measurement interval (how often the scanner collected data) from the file
    CF0 = float(twix[0]['hdr']['Dicom']['lFrequency'])             # Retrieved the main frequency used for the scan from the file header
    im = np.squeeze(np.asarray(twix[0]['image'][:]))               # Extracted the measurement (image) data and formatted it for analysis
    if im.ndim < 2:                                                # Checked for missing or incomplete measurement data, and skipped the file with a warning if found
        st.warning(f"❌ Not enough usable image data → {os.path.basename(dat_file)}")  
        return None, None, None                                    


    # reshape real/imag into complex [coil, freq]
    if im.shape[-2] == 2:                                        # Checked if the second-to-last dimension is 2 (means there are both real and imaginary data channels)
        coils = int(np.prod(im.shape[:-2]))                      # Counted the number of coils (detectors) in the measurement
        _, i_q, nfreq = im.shape                                 # Unpacked the image shape (i_q should always be 2: real and imaginary)
        im_resh = im.reshape((coils, i_q, nfreq))                # Rearranged the image data into (coil, channel, frequency) format
        ppmksp = im_resh[:,0,:] + 1j*im_resh[:,1,:]              # Combined real and imaginary data to form a complex-valued dataset, required for further calculations
    else:
        st.warning(f"❌ Unexpected shape {im.shape}")             # Displayed a warning if the data format was not as expected (might indicate a corrupted or unsupported file)
        return None, None, None                                  # Stopped processing this file, as it could not be used


    # IFFT + coil combine
    ppmfft = np.abs(np.fft.fftshift(np.fft.ifft(ppmksp, axis=1), axes=1))         # Converted the signal from the time domain to the frequency domain for each coil using the inverse FFT (ifft), then centered the zero frequency in the result using fftshift for easier visualization
    ppmdata = np.sqrt((ppmfft**2).sum(axis=0))                                    # Combined all coil signals into one overall signal by taking the square root of the sum of the squared values (a standard MRI approach)
    ppmdata /= ppmdata.max()                                                      # Normalized the combined signal so its maximum value became 1

    # Frequency axis
    N = ppmdata.size                                                              # Counted the number of points in the spectrum (number of frequency bins)
    bw_per_px = 1e9 / (dwelltime * N)                                             # Calculated the frequency width of each data point, based on dwell time and number of points
    SW = bw_per_px * N                                                            # Calculated the total frequency range (sweep width) covered by the scan
    xvals = np.linspace(-SW/2, SW/2, N)                                           # Generated an array of frequency values (in Hz) to use for the x-axis of the plot

    # Peak & half-max
    peakIdx = int(np.argmax(ppmdata))                                             # Found the index of the main signal peak in the combined spectrum
    peakVal = ppmdata[peakIdx]                                                    # Retrieved the value of the signal at the peak
    halfVal = peakVal / 2                                                         # Calculated half of the peak value to use for measuring the width of the peak (FWHM)

    # Compute CF shift
    delta_cf = round((peakIdx - (N/2)) * bw_per_px)                               # Measured how far the main peak had shifted from the center of the spectrum, in Hz
    CF_new = CF0 + delta_cf                                                       # Updated the center frequency to account for any shift in the main signal peak


    # FWHM interpolation
    xpL    = ppmdata[:peakIdx+1]                                               # Selected all signal values from the start up to the peak (left side of the peak)
    fpL    = xvals[:peakIdx+1]                                                 # Selected the corresponding frequency values for the left side
    xLeft  = np.interp(halfVal, xpL, fpL)                                      # Found the frequency where the signal first dropped to half its maximum on the left

    xpR        = ppmdata[peakIdx:]                                              # Selected all signal values from the peak to the end (right side)
    fpR        = xvals[peakIdx:]                                                # Selected the corresponding frequency values for the right side
    xRight     = np.interp(halfVal, xpR[::-1], fpR[::-1])                       # Found the frequency where the signal dropped to half its maximum on the right

    # FWHM in Hz & ppm
    FWHM_Hz   = xRight - xLeft                                                  # Calculated the full width at half maximum (FWHM) in Hz by subtracting left from right crossing
    FWHM_ppm  = FWHM_Hz / CF_new * 1e6                                          # Converted the FWHM from Hz to parts per million (ppm) for standard MRI reporting

    # prepare title & output filename
    base     = os.path.basename(dat_file)                                       # Extracted the file name from the full file path
    GA_label = base.split("_")[-1].replace(".dat","")                           # Extracted the gradient angle label from the file name
    title    = f"FID GA = {GA_label}"                                           # Created the title for the plot using the gradient angle


    # ─── Plot spectrum ─────────────────────────────────────────────────────────
    plt.figure(figsize=(12,6))                                                     # Created a new plot window sized 12x6 inches
    plt.plot(xvals, ppmdata, 'b', linewidth=1.2)                                   # Plotted the frequency spectrum as a blue line
    plt.hlines(halfVal, xLeft, xRight, colors='k', linewidth=5)                    # Drew a thick black line at half-max value to show the FWHM

    plt.xlim(-500, 500)                                                            # Set the x-axis to show frequencies from -500 to 500 Hz
    plt.ylim(0, 1.05)                                                              # Set the y-axis to show normalized amplitude from 0 to just above 1
    plt.xlabel("Frequency [Hz]")                                                    # Added a label for the x-axis
    plt.ylabel("Normalized Amplitude")                                              # Added a label for the y-axis
    plt.title(title, fontsize=14, fontweight='bold')                                # Added a bold title using the gradient angle label

    txtX = 140                                                                     # Set the horizontal position for plot annotations
    plt.text(txtX, 0.85, f"CF   = {CF_new:.0f} Hz",  fontweight='bold')            # Annotated the new center frequency on the plot
    plt.text(txtX, 0.75, f"FWHM = {FWHM_Hz:.2f} Hz", fontweight='bold')            # Annotated the FWHM in Hz
    plt.text(txtX, 0.65, f"FWHM = {FWHM_ppm:.2f} ppm",fontweight='bold')           # Annotated the FWHM in ppm
    plt.text(txtX, 0.35, "GO$_x$ = uT/m or DAQ")                                   # Added an extra annotation for x-gradient info
    plt.text(txtX, 0.25, "GO$_y$ = uT/m or DAQ")                                   # Added an extra annotation for y-gradient info
    plt.text(txtX, 0.15, "GO$_z$ = uT/m or DAQ")                                   # Added an extra annotation for z-gradient info

    plt.grid(True)                                                                 # Displayed a grid on the plot for easier reading
    plt.savefig(os.path.join(plotsdir, f"{base}.jpg"), dpi=150, bbox_inches='tight') # Saved the plot as a JPG image in the output folder
    plt.close()                                                                    # Closed the plot window to free memory

    return [CF_new, FWHM_Hz, FWHM_ppm], ppmdata, GA_label                          # Returned the results, processed signal, and label for further use

# ─── Streamlit App ────────────────────────────────────────────────────────────
st.title(".DAT Processor")                                                  # Set the web app title in the Streamlit interface

uploaded_files = st.file_uploader(
    "Drop or browse .dat files here",                                       # Displayed a file uploader for the user to upload .dat files
    type=["dat"],                                                           # Limited uploads to only .dat files
    accept_multiple_files=True                                              # Allowed the user to select and upload multiple files at once
)

if uploaded_files:                                                          # Checked if the user had uploaded any files
    # make temp QA folders
    datdir   = tempfile.mkdtemp()                                           # Created a temporary directory for storing uploaded files and results
    qadir    = os.path.join(datdir, "QA")                                   # Set the path for the main QA folder
    plotsdir = os.path.join(qadir, "PLOTS_PPM")                             # Set the path for saving plots
    exceldir = os.path.join(qadir, "Excels")                                # Set the path for saving Excel files
    for d in (qadir, plotsdir, exceldir):
        os.makedirs(d, exist_ok=True)                                       # Made sure each output folder existed (created them if they did not)

    ppm_summary  = []                                                       # Initialized an empty list to store summary information for each file
    fft_data_all = []                                                       # Initialized an empty list to store processed FFT data for each file
    GA_labels    = []                                                       # Initialized an empty list to store gradient angle labels
    nx = 0                                                                  # Initialized variable to store the number of FFT data points


    # process each file
    for f in uploaded_files:                                                    # Went through each uploaded .dat file
        local_file = os.path.join(datdir, f.name)                               # Set the local path to temporarily save this file
        with open(local_file, "wb") as out:
            out.write(f.read())                                                 # Saved the uploaded file’s content to disk
        st.write(f"📂 Processing: {f.name}")                                    # Showed a message to the user about which file is being processed
        summary, fftdata, ga = process_dat_file(local_file, plotsdir)           # Ran the analysis function on the current file
        if summary:                                                             # If analysis was successful for this file:
            ppm_summary.append(summary)                                         #   Added summary results to the list
            fft_data_all.append(fftdata)                                        #   Added FFT spectrum data to the list
            GA_labels.append(ga)                                                #   Added the gradient angle label to the list
            nx = fftdata.size                                                   #   Recorded the number of FFT points

    if ppm_summary:                                                             # Checked if at least one file was successfully processed
        # Excel outputs
        df_sum = pd.DataFrame(ppm_summary, columns=["CF", "FWHM Hz", "FWHM ppm"])   # Created a summary table (DataFrame) for center freq and FWHM for all files
        df_sum.insert(0, "GA_Label", GA_labels)                                     # Inserted GA labels as the first column

        df_fft = pd.DataFrame(np.array(fft_data_all).T, columns=GA_labels)           # Created a table of FFT spectra, each column is a GA file
        df_fft.insert(0, "SampleIdx", np.arange(1, nx+1))                            # Added a column for sample index numbers

        excel_path = os.path.join(exceldir, "QA_Results.xlsx")                       # Set the path for the Excel output file
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_sum.to_excel(writer, sheet_name="PPM_Summary", index=False)           # Saved the summary table to an Excel sheet
            df_fft.to_excel(writer, sheet_name="PPM_Info_full", index=False)         # Saved the FFT table to another sheet


        # Radar/polar plot (no legends)
        angle_deg       = np.array([float(lbl) for lbl in GA_labels])                          # Converted the GA labels to numeric angle values (degrees)
        idx             = np.argsort(angle_deg)                                                # Got the order needed to sort the data by angle
        angle_deg_sorted= angle_deg[idx]                                                       # Sorted the angles in ascending order
        ppm_sorted      = df_sum["FWHM ppm"].values[idx]                                       # Reordered the FWHM (ppm) data to match the sorted angles
        theta           = np.deg2rad(np.append(angle_deg_sorted, angle_deg_sorted[0]))         # Converted the sorted angles to radians and repeated the first value at the end to close the curve
        r_ppm           = np.append(ppm_sorted, ppm_sorted[0])                                 # Closed the curve for the plot by repeating the first FWHM value at the end
        bound_ppm       = 5.0 * np.ones_like(theta)                                            # Created a circular boundary at 5 ppm for visual reference

        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'polar':True})                       # Created a new polar (radar) plot with size 8x8 inches
        ax.plot(theta, r_ppm, 'k-', linewidth=1)                                               # Plotted the FWHM curve as a black line
        ax.plot(theta, bound_ppm, 'r-', linewidth=1)                                           # Plotted the reference boundary as a red circle
        ax.plot(theta[:-1], r_ppm[:-1], 'g*', markersize=8)                                    # Plotted green stars at each data point

        ax.set_title("HOMOGENEITY v. GA", fontweight='bold', y=1.10)                           # Added a bold title to the plot
        ax.set_theta_zero_location('E')                                                        # Set zero degrees to point right (East)
        ax.set_theta_direction(1)                                                              # Set angle increase direction to counterclockwise
        ax.set_rlim(0, 10)                                                                     # Set the radial axis to go from 0 to 10
        ax.set_rticks(np.arange(0, 11, 2))                                                     # Set radial ticks every 2 units
        ax.set_thetagrids(angle_deg_sorted)                                                    # Labeled the angular axis with the GA angle values

        fig.savefig(os.path.join(plotsdir, "Homogeneity_vs_GA_POLAR_PPM.png"),
                    dpi=150, bbox_inches='tight')                                              # Saved the radar plot as an image in the results folder
        plt.close(fig)                                                                         # Closed the plot to free up memory

        # zip & download
        buf = BytesIO()                                                         # Created an in-memory buffer to temporarily hold the zip file data
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:             # Opened a new zip file for writing (compressed format)
            for root, _, files in os.walk(qadir):                               # Walked through all folders and files in the QA results directory
                for fn in files:                                                # For each file found:
                    full = os.path.join(root, fn)                               #   Got the full path of the file
                    zf.write(full, os.path.relpath(full, qadir))                #   Added the file to the zip, preserving the folder structure
        buf.seek(0)                                                             # Rewound the buffer to the beginning so it could be read for download
        st.download_button(
            "📥 Download QA ZIP",                                               # Displayed a download button labeled with an icon and text
            data=buf,                                                          # Used the in-memory buffer as the file to download
            file_name="QA_Results.zip",                                        # Named the downloaded file
            mime="application/zip"                                             # Specified the file type as a zip archive
        )
        st.success("✅ Done! Download your QA bundle above.")                   # Displayed a success message after the zip was ready
    else:
        st.warning("⚠️ No valid files processed.")                             # Warned the user if no valid data files were analyzed
