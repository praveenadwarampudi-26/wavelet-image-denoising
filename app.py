
import io
import time

import numpy as np
import pywt
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="WaveletLab | Image Denoising",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM DESIGN - CSS ONLY
# No HTML dashboard content is used.
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #0B1020;
    color: #F1F5FF;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: #0B1020;
}

[data-testid="stSidebar"] {
    background: #10192C;
    border-right: 1px solid #293B59;
}

[data-testid="stSidebar"] * {
    color: #F1F5FF;
}

[data-testid="stSidebar"] label {
    color: #C3CFE4;
}

section.main > div.block-container {
    max-width: 1450px;
    padding: 2rem 3rem 3rem;
}

h1, h2, h3 {
    color: #F1F5FF !important;
    font-weight: 700 !important;
}

p {
    color: #B8C5DA;
    line-height: 1.7;
}

.hero {
    background: linear-gradient(120deg, #192B50, #28204F, #12364B);
    border: 1px solid #3B5078;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 28px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #C8D5EA;
    font-size: 15px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #121C32;
    border: 1px dashed #64789E;
    border-radius: 13px;
}

div[data-testid="stFileUploaderDropzone"] * {
    color: #EAF0FF !important;
}

div[data-testid="stFileUploaderDropzone"] button {
    background: #293B59 !important;
    color: white !important;
    border: 1px solid #536B91 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #17243B;
    color: white;
    border-color: #344968;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #7656E8, #388DD9);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border: 1px solid #8BE6FF;
    box-shadow: 0 0 18px #7656E844;
}

div[data-testid="stMetric"] {
    background: #121C32;
    border: 1px solid #293B59;
    border-radius: 14px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #AAB9D2;
}

div[data-testid="stMetricValue"] {
    color: #65D9F5;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #344968;
}

hr {
    border-color: #293B59;
}

.footer {
    text-align: center;
    color: #899BB9;
    font-size: 12px;
    padding: 25px 5px;
    border-top: 1px solid #293B59;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# IMAGE PROCESSING FUNCTIONS
# ==========================================================

def add_gaussian_noise(image, noise_level):
    rng = np.random.default_rng(42)

    noise = rng.normal(
        0,
        noise_level,
        image.shape
    )

    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def denoise_channel(channel, wavelet_name, level, threshold_mode):

    channel = channel.astype(np.float64)

    wavelet = pywt.Wavelet(wavelet_name)

    max_level = pywt.dwt_max_level(
        min(channel.shape),
        wavelet.dec_len
    )

    if max_level < 1:
        return channel.astype(np.uint8)

    actual_level = min(level, max_level)

    coefficients = pywt.wavedec2(
        channel,
        wavelet_name,
        level=actual_level
    )

    # Estimate noise from finest diagonal coefficients
    diagonal = coefficients[-1][2]

    sigma = np.median(
        np.abs(diagonal)
    ) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(
        2 * np.log(channel.size)
    )

    filtered_coefficients = [coefficients[0]]

    for detail_level in coefficients[1:]:

        filtered_details = tuple(
            pywt.threshold(
                band,
                threshold,
                mode=threshold_mode
            )
            for band in detail_level
        )

        filtered_coefficients.append(filtered_details)

    reconstructed = pywt.waverec2(
        filtered_coefficients,
        wavelet_name
    )

    reconstructed = reconstructed[
        :channel.shape[0],
        :channel.shape[1]
    ]

    return np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)


def wavelet_denoise(image, wavelet_name, level, threshold_mode):

    if image.ndim == 2:
        return denoise_channel(
            image,
            wavelet_name,
            level,
            threshold_mode
        )

    output_channels = []

    for channel_index in range(image.shape[2]):

        restored_channel = denoise_channel(
            image[:, :, channel_index],
            wavelet_name,
            level,
            threshold_mode
        )

        output_channels.append(restored_channel)

    return np.stack(output_channels, axis=2)


def calculate_metrics(reference, test_image):

    mse = mean_squared_error(
        reference,
        test_image
    )

    psnr = peak_signal_noise_ratio(
        reference,
        test_image,
        data_range=255
    )

    return mse, psnr


def image_to_bytes(image):

    output = io.BytesIO()

    Image.fromarray(image).save(
        output,
        format="PNG"
    )

    return output.getvalue()


def create_wavelet_visualization(image, wavelet_name):

    grayscale = np.array(
        Image.fromarray(image).convert("L")
    )

    approximation, details = pywt.dwt2(
        grayscale.astype(np.float64),
        wavelet_name
    )

    horizontal, vertical, diagonal = details

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14, 4)
    )

    components = [
        (approximation, "Approximation (LL)"),
        (horizontal, "Horizontal (LH)"),
        (vertical, "Vertical (HL)"),
        (diagonal, "Diagonal (HH)")
    ]

    for axis, (component, title) in zip(axes, components):

        axis.imshow(
            component,
            cmap="gray"
        )

        axis.set_title(
            title,
            color="white",
            fontsize=10
        )

        axis.axis("off")

    fig.patch.set_facecolor("#121C32")

    plt.tight_layout()

    return fig


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🌊 WaveletLab")

    st.caption("IMAGE PROCESSING LABORATORY")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        [
            "Dashboard",
            "Methodology",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("⚙️ Processing Settings")

    wavelet_name = st.selectbox(
        "Wavelet Transform",
        ["haar", "db2", "db4", "sym4", "coif1"]
    )

    level = st.slider(
        "Decomposition Level",
        1,
        5,
        2
    )

    threshold_mode = st.selectbox(
        "Thresholding Technique",
        ["soft", "hard"]
    )

    noise_level = st.slider(
        "Gaussian Noise Intensity",
        0,
        60,
        25
    )

    st.divider()

    st.caption("DIGITAL IMAGE PROCESSING PROJECT")
    st.caption("Wavelet-based image restoration")


# ==========================================================
# MAIN HEADER - NATIVE STREAMLIT CONTENT
# ==========================================================

st.markdown('<div class="hero">', unsafe_allow_html=True)

st.caption("DIP LABORATORY  /  IMAGE RESTORATION")

st.markdown(
    '<div class="hero-title">Wavelet Image Denoising</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'An intelligent image restoration platform using '
    'Discrete Wavelet Transform and thresholding techniques.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# DASHBOARD PAGE
# ==========================================================

if page == "Dashboard":

    st.title("Image Processing Dashboard")

    st.caption(
        "Upload an image, configure parameters and perform image restoration."
    )

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["png", "jpg", "jpeg", "bmp", "tif"]
    )

    if uploaded_file is not None:

        original = np.array(
            Image.open(uploaded_file).convert("RGB")
        ).astype(np.uint8)

        noisy = add_gaussian_noise(
            original,
            noise_level
        )

        current_settings = (
            uploaded_file.name,
            wavelet_name,
            level,
            threshold_mode,
            noise_level
        )

        # Clear previous results when the input or settings change
        if st.session_state.get("result_settings") != current_settings:
            st.session_state.pop("wavelet_result", None)

        st.subheader("Experiment Configuration")

        c1, c2, c3 = st.columns(3)

        c1.metric("Wavelet", wavelet_name.upper())
        c2.metric("Thresholding", threshold_mode.capitalize())
        c3.metric("Noise Intensity", f"{noise_level} σ")

        st.subheader("Input Image Preview")

        left, right = st.columns(2)

        with left:
            st.image(
                original,
                caption="Original Image",
                use_container_width=True
            )

        with right:
            st.image(
                noisy,
                caption="Image with Gaussian Noise",
                use_container_width=True
            )

        if st.button(
            "🚀 RUN WAVELET DENOISING",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Processing image using Wavelet Transform..."
                ):

                    start_time = time.perf_counter()

                    denoised = wavelet_denoise(
                        noisy,
                        wavelet_name,
                        level,
                        threshold_mode
                    )

                    processing_time = (
                        time.perf_counter() - start_time
                    )

                    mse_noisy, psnr_noisy = calculate_metrics(
                        original,
                        noisy
                    )

                    mse_denoised, psnr_denoised = calculate_metrics(
                        original,
                        denoised
                    )

                    st.session_state["wavelet_result"] = {
                        "original": original,
                        "noisy": noisy,
                        "denoised": denoised,
                        "mse_noisy": mse_noisy,
                        "psnr_noisy": psnr_noisy,
                        "mse_denoised": mse_denoised,
                        "psnr_denoised": psnr_denoised,
                        "processing_time": processing_time
                    }

                    st.session_state["result_settings"] = current_settings

            except Exception as error:
                st.error(f"Processing failed: {error}")

        # Display results
        if "wavelet_result" in st.session_state:

            result = st.session_state["wavelet_result"]

            st.success("Image denoising completed successfully!")

            st.title("Restoration Results")

            a, b, c = st.columns(3)

            with a:
                st.image(
                    result["original"],
                    caption="Original Image",
                    use_container_width=True
                )

            with b:
                st.image(
                    result["noisy"],
                    caption="Noisy Image",
                    use_container_width=True
                )

            with c:
                st.image(
                    result["denoised"],
                    caption="Denoised Image",
                    use_container_width=True
                )

            st.title("Performance Analysis")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "PSNR After Denoising",
                f"{result['psnr_denoised']:.2f} dB",
                delta=f"{result['psnr_denoised'] - result['psnr_noisy']:.2f} dB"
            )

            m2.metric(
                "MSE After Denoising",
                f"{result['mse_denoised']:.2f}",
                delta=f"{result['mse_denoised'] - result['mse_noisy']:.2f}",
                delta_color="inverse"
            )

            m3.metric(
                "Processing Time",
                f"{result['processing_time']:.4f} sec"
            )

            st.title("Wavelet Decomposition Analysis")

            st.caption(
                "Visualisation of approximation and detail coefficients."
            )

            try:

                figure = create_wavelet_visualization(
                    result["original"],
                    wavelet_name
                )

                st.pyplot(
                    figure,
                    use_container_width=True
                )

                plt.close(figure)

            except Exception as error:
                st.warning(f"Could not create decomposition plot: {error}")

            st.title("Download Restored Image")

            st.download_button(
                "⬇️ DOWNLOAD DENOISED IMAGE",
                data=image_to_bytes(result["denoised"]),
                file_name="wavelet_denoised.png",
                mime="image/png",
                use_container_width=True
            )

            with st.expander("View Detailed Experiment Report"):

                st.write({
                    "Wavelet": wavelet_name,
                    "Decomposition Level": level,
                    "Thresholding": threshold_mode,
                    "Noise Intensity": noise_level,
                    "Noisy Image MSE": round(result["mse_noisy"], 4),
                    "Denoised Image MSE": round(result["mse_denoised"], 4),
                    "Noisy Image PSNR": round(result["psnr_noisy"], 4),
                    "Denoised Image PSNR": round(result["psnr_denoised"], 4),
                    "Processing Time": round(result["processing_time"], 6)
                })

    else:

        st.info(
            "Upload an image to begin your wavelet denoising experiment."
        )

        st.subheader("How It Works")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 01 · Upload")
            st.write("Choose a digital image for processing.")

        with col2:
            st.markdown("### 02 · Configure")
            st.write("Select the wavelet and thresholding parameters.")

        with col3:
            st.markdown("### 03 · Restore")
            st.write("Remove simulated noise and analyse the results.")


# ==========================================================
# METHODOLOGY PAGE
# ==========================================================

elif page == "Methodology":

    st.title("Project Methodology")

    st.write(
        "Wavelet-based image denoising separates an image into "
        "frequency components. Thresholding is applied to detail "
        "coefficients to reduce noise while preserving image features."
    )

    steps = [
        (
            "01. Image Acquisition",
            "Upload the original digital image."
        ),
        (
            "02. Noise Addition",
            "Introduce simulated Gaussian noise."
        ),
        (
            "03. Wavelet Decomposition",
            "Apply DWT to obtain approximation and detail coefficients."
        ),
        (
            "04. Thresholding",
            "Suppress noise-related coefficients using soft or hard thresholding."
        ),
        (
            "05. Image Reconstruction",
            "Apply inverse wavelet transform to reconstruct the restored image."
        ),
        (
            "06. Performance Evaluation",
            "Compare the original and restored images using PSNR and MSE."
        )
    ]

    for heading, description in steps:

        with st.container(border=True):
            st.subheader(heading)
            st.write(description)

    st.title("Mathematical Equations")

    st.subheader("Universal Threshold")

    st.latex(
        r"T = \sigma\sqrt{2\ln(N)}"
    )

    st.write(
        "Sigma represents estimated noise standard deviation; "
        "N represents the number of image samples."
    )

    st.subheader("Mean Squared Error")

    st.latex(
        r"MSE = \frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}"
        r"(I(i,j)-K(i,j))^2"
    )

    st.subheader("Peak Signal-to-Noise Ratio")

    st.latex(
        r"PSNR = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)"
    )

    st.subheader("Discrete Wavelet Transform")

    st.write(
        "DWT decomposes an image into four components: "
        "approximation (LL), horizontal detail (LH), "
        "vertical detail (HL), and diagonal detail (HH)."
    )


# ==========================================================
# ABOUT PROJECT PAGE
# ==========================================================

elif page == "About Project":

    st.title("About WaveletLab")

    st.subheader("Project Objective")

    st.write(
        "The objective of this project is to reduce unwanted "
        "noise from digital images while preserving important "
        "image structures using Wavelet Transform and coefficient "
        "thresholding."
    )

    st.divider()

    st.subheader("Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Python
        - Streamlit
        - NumPy
        - PyWavelets
        """)

    with col2:
        st.markdown("""
        - Pillow
        - Matplotlib
        - Scikit-image
        - Discrete Wavelet Transform
        """)

    st.divider()

    st.subheader("Wavelet Families")

    st.write("""
    - **Haar:** Simple and computationally efficient.
    - **Daubechies:** Includes db2 and db4 wavelets.
    - **Symlet:** Near-symmetric wavelet family.
    - **Coiflet:** Wavelet family with additional vanishing moments.
    """)

    st.subheader("Applications")

    st.write("""
    - Medical image enhancement
    - Satellite image processing
    - Digital photography
    - Image restoration
    - Computer vision
    """)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">
    WAVELETLAB · DIGITAL IMAGE PROCESSING
    <br>
    Wavelet-Based Image Restoration System
</div>
""", unsafe_allow_html=True)
