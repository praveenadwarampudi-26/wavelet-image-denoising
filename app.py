
import io
import time

import numpy as np
import pywt
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="WaveletLab | Image Restoration",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B1020;
    --panel: #121C32;
    --border: #293B59;
    --muted: #9AAAC5;
    --white: #F1F5FF;
    --purple: #9B87FF;
    --cyan: #65D9F5;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: var(--bg);
    color: var(--white);
}

[data-testid="stHeader"] {
    background: rgba(11,16,32,0.95);
}

[data-testid="stSidebar"] {
    background: #10192C;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--white) !important;
}

[data-testid="stSidebar"] label {
    color: #C0CCE0 !important;
}

section.main > div.block-container {
    max-width: 1500px;
    padding: 2rem 3rem 3rem;
}

h1, h2, h3, h4 {
    color: var(--white) !important;
    letter-spacing: -0.5px;
}

p {
    color: #B8C5DA;
    line-height: 1.7;
}

.hero {
    background: linear-gradient(115deg, #192B50, #28204F 55%, #12364B);
    border: 1px solid #3B5078;
    border-radius: 20px;
    padding: 32px 35px;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 12px 0;
    color: white !important;
}

.hero p {
    color: #CBD7EB;
    max-width: 800px;
    font-size: 15px;
}

.eyebrow {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.panel h3 {
    margin-top: 0;
}

.small-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #121C32 !important;
    border: 1px dashed #61749A !important;
    border-radius: 14px !important;
}

div[data-testid="stFileUploaderDropzone"] * {
    color: #E5ECFF !important;
}

div[data-testid="stFileUploaderDropzone"] button {
    background: #273653 !important;
    border: 1px solid #53678E !important;
    color: white !important;
    border-radius: 8px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #17243B !important;
    color: white !important;
    border: 1px solid #344968 !important;
    border-radius: 9px !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #7656E8, #388DD9);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 700;
    min-height: 45px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border: 1px solid #8BE6FF;
    box-shadow: 0 0 20px #7656E844;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #AAB9D2;
}

div[data-testid="stMetricValue"] {
    color: var(--cyan);
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #344968;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
}

.stTabs [data-baseweb="tab"] {
    color: #AAB9D2;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #9B87FF !important;
}

hr {
    border-color: #293B58;
}

.footer {
    text-align: center;
    color: #899BB9;
    font-size: 12px;
    padding: 30px 10px;
    border-top: 1px solid #25334D;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# IMAGE PROCESSING FUNCTIONS
# =========================================================

def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise to an image."""

    rng = np.random.default_rng(42)

    noise = rng.normal(
        0,
        noise_level,
        image.shape
    )

    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def denoise_channel(channel, wavelet_name, level, threshold_mode):
    """Denoise one image channel using wavelet thresholding."""

    channel = channel.astype(np.float64)

    wavelet = pywt.Wavelet(wavelet_name)

    max_level = pywt.dwt_max_level(
        min(channel.shape),
        wavelet.dec_len
    )

    if max_level < 1:
        return channel.astype(np.uint8)

    actual_level = min(level, max_level)

    coeffs = pywt.wavedec2(
        channel,
        wavelet_name,
        level=actual_level
    )

    # Estimate noise using diagonal detail coefficients
    diagonal = coeffs[-1][2]

    sigma = np.median(
        np.abs(diagonal)
    ) / 0.6745

    # Universal threshold
    threshold = sigma * np.sqrt(
        2 * np.log(channel.size)
    )

    filtered_coeffs = [coeffs[0]]

    for detail_level in coeffs[1:]:

        filtered_details = tuple(
            pywt.threshold(
                band,
                threshold,
                mode=threshold_mode
            )
            for band in detail_level
        )

        filtered_coeffs.append(filtered_details)

    reconstructed = pywt.waverec2(
        filtered_coeffs,
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
    """Denoise grayscale or RGB images."""

    if image.ndim == 2:
        return denoise_channel(
            image,
            wavelet_name,
            level,
            threshold_mode
        )

    channels = []

    for i in range(image.shape[2]):

        result = denoise_channel(
            image[:, :, i],
            wavelet_name,
            level,
            threshold_mode
        )

        channels.append(result)

    return np.stack(channels, axis=2)


def calculate_metrics(reference, test_image):
    """Calculate MSE and PSNR."""

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
    """Convert image into downloadable PNG bytes."""

    output = io.BytesIO()

    Image.fromarray(image).save(
        output,
        format="PNG"
    )

    return output.getvalue()


def create_wavelet_visualization(image, wavelet_name):
    """Visualize approximation and detail coefficients."""

    grayscale = np.array(
        Image.fromarray(image).convert("L")
    )

    coeffs = pywt.dwt2(
        grayscale.astype(np.float64),
        wavelet_name
    )

    approximation, (horizontal, vertical, diagonal) = coeffs

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14, 4)
    )

    images = [
        (approximation, "Approximation"),
        (horizontal, "Horizontal Detail"),
        (vertical, "Vertical Detail"),
        (diagonal, "Diagonal Detail")
    ]

    for ax, (data, title) in zip(axes, images):

        ax.imshow(
            data,
            cmap="gray"
        )

        ax.set_title(
            title,
            color="white",
            fontsize=9
        )

        ax.axis("off")

    fig.patch.set_facecolor("#121C32")

    plt.tight_layout()

    return fig


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌊 WaveletLab")

    st.caption("IMAGE PROCESSING LABORATORY")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Choose Section",
        [
            "Dashboard",
            "Methodology",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### ⚙️ Processing Settings")

    wavelet_name = st.selectbox(
        "Wavelet Transform",
        ["haar", "db2", "db4", "sym4", "coif1"]
    )

    level = st.slider(
        "Decomposition Level",
        min_value=1,
        max_value=5,
        value=2
    )

    threshold_mode = st.selectbox(
        "Thresholding Technique",
        ["soft", "hard"]
    )

    noise_level = st.slider(
        "Gaussian Noise Intensity",
        min_value=0,
        max_value=60,
        value=25
    )

    st.divider()

    st.caption("DIGITAL IMAGE PROCESSING PROJECT")
    st.caption("Wavelet-based image restoration")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="eyebrow">
        DIP LABORATORY / IMAGE RESTORATION
    </div>

    <h1>WaveletLab</h1>

    <p>
        Advanced digital image restoration using
        Discrete Wavelet Transform. Analyse, restore
        and evaluate image quality through intelligent
        coefficient thresholding.
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("## Image Processing Dashboard")

    st.caption(
        "Upload an image and configure your denoising experiment."
    )

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["png", "jpg", "jpeg", "bmp", "tif"]
    )

    if uploaded_file:

        original = np.array(
            Image.open(uploaded_file).convert("RGB")
        )

        original = original.astype(np.uint8)

        noisy = add_gaussian_noise(
            original,
            noise_level
        )

        st.markdown("### Experiment Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="panel">
                <div class="small-label">WAVELET</div>
                <h3>"""+wavelet_name.upper()+"""</h3>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="panel">
                <div class="small-label">THRESHOLDING</div>
                <h3>"""+threshold_mode.capitalize()+"""</h3>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="panel">
                <div class="small-label">NOISE INTENSITY</div>
                <h3>"""+str(noise_level)+""" σ</h3>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Input Image Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                noisy,
                caption="Simulated Gaussian Noise",
                use_container_width=True
            )

        st.markdown("")

        if st.button(
            "🚀 RUN WAVELET DENOISING",
            use_container_width=True
        ):

            with st.spinner(
                "Analysing wavelet coefficients and restoring image..."
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
                    "time": processing_time,
                    "wavelet": wavelet_name,
                    "level": level,
                    "threshold": threshold_mode,
                    "noise": noise_level
                }

        if "wavelet_result" in st.session_state:

            result = st.session_state["wavelet_result"]

            # Do not show old results after settings or image change
            current_settings = (
                uploaded_file.name,
                wavelet_name,
                level,
                threshold_mode,
                noise_level
            )

            if st.session_state.get("result_settings") != current_settings:
                st.session_state.pop("wavelet_result", None)

            else:

                denoised = result["denoised"]

                st.success(
                    "Image denoising completed successfully!"
                )

                st.markdown("## Restoration Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        result["original"],
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        result["noisy"],
                        caption="Noisy Image",
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        denoised,
                        caption="Denoised Image",
                        use_container_width=True
                    )

                st.markdown("## Performance Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "PSNR After Denoising",
                        f"{result['psnr_denoised']:.2f} dB",
                        delta=f"{result['psnr_denoised'] - result['psnr_noisy']:.2f} dB"
                    )

                with col2:
                    st.metric(
                        "MSE After Denoising",
                        f"{result['mse_denoised']:.2f}",
                        delta=f"{result['mse_denoised'] - result['mse_noisy']:.2f}",
                        delta_color="inverse"
                    )

                with col3:
                    st.metric(
                        "Processing Time",
                        f"{result['time']:.4f} sec"
                    )

                st.markdown("## Wavelet Decomposition Analysis")

                st.caption(
                    "First-level decomposition into approximation "
                    "and horizontal, vertical and diagonal details."
                )

                fig = create_wavelet_visualization(
                    result["original"],
                    result["wavelet"]
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                st.markdown("## Download Restored Image")

                st.download_button(
                    label="⬇️ DOWNLOAD DENOISED IMAGE",
                    data=image_to_bytes(denoised),
                    file_name="wavelet_denoised.png",
                    mime="image/png",
                    use_container_width=True
                )

                with st.expander("View Detailed Experiment Report"):

                    st.write({
                        "Wavelet": result["wavelet"],
                        "Decomposition Level": result["level"],
                        "Thresholding": result["threshold"],
                        "Gaussian Noise": result["noise"],
                        "Noisy Image MSE": round(result["mse_noisy"], 4),
                        "Denoised Image MSE": round(result["mse_denoised"], 4),
                        "Noisy Image PSNR": round(result["psnr_noisy"], 4),
                        "Denoised Image PSNR": round(result["psnr_denoised"], 4),
                        "Processing Time (seconds)": round(result["time"], 6)
                    })

    else:

        st.markdown("""
        <div class="panel">

            <h3>📂 Ready to Process</h3>

            <p>
                Upload your digital image to begin.
                Select your wavelet transform and configure
                the denoising parameters in the sidebar.
            </p>

            <p>
                The system will simulate Gaussian noise,
                perform wavelet-based denoising and evaluate
                restoration performance.
            </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# METHODOLOGY
# =========================================================

elif page == "Methodology":

    st.markdown("## Project Methodology")

    st.write(
        "Wavelet-based image denoising is a signal-processing "
        "technique that separates an image into frequency "
        "components and suppresses noise using thresholding."
    )

    steps = [
        (
            "01",
            "Image Acquisition",
            "Upload the original digital image."
        ),
        (
            "02",
            "Noise Addition",
            "Introduce simulated Gaussian noise."
        ),
        (
            "03",
            "Wavelet Decomposition",
            "Apply the Discrete Wavelet Transform to separate approximation and detail coefficients."
        ),
        (
            "04",
            "Thresholding",
            "Suppress noise-related detail coefficients using soft or hard thresholding."
        ),
        (
            "05",
            "Image Reconstruction",
            "Apply the inverse wavelet transform to reconstruct the restored image."
        ),
        (
            "06",
            "Performance Evaluation",
            "Compare the restored image with the original using PSNR and MSE."
        )
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="panel">

            <span style="color:#65D9F5;font-weight:800">
                STEP {number}
            </span>

            <h3>{title}</h3>

            <p>{description}</p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Mathematical Equations")

    st.markdown("### Universal Threshold")

    st.latex(
        r"T = \sigma \sqrt{2\ln(N)}"
    )

    st.write(
        "Where sigma is the estimated noise standard deviation "
        "and N is the number of image samples."
    )

    st.markdown("### Mean Squared Error")

    st.latex(
        r"MSE = \frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}"
        r"(I(i,j)-K(i,j))^2"
    )

    st.markdown("### Peak Signal-to-Noise Ratio")

    st.latex(
        r"PSNR = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)"
    )

    st.markdown("### Discrete Wavelet Transform")

    st.write(
        "The DWT decomposes an image into an approximation "
        "component and three detail components: horizontal, "
        "vertical and diagonal."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.markdown("## About WaveletLab")

    st.markdown("""
    <div class="panel">

        <h3>Project Objective</h3>

        <p>
            The objective of this project is to reduce unwanted
            noise from digital images while preserving important
            image structures and details using Wavelet Transform
            and coefficient thresholding.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="panel">
            <h3>Programming</h3>
            <p>Python</p>
            <p>NumPy</p>
            <p>Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="panel">
            <h3>Image Processing</h3>
            <p>PyWavelets</p>
            <p>Matplotlib</p>
            <p>Scikit-image</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Available Wavelet Families")

    st.write("""
    - Haar: Simple and computationally efficient.
    - Daubechies (db2, db4): Compactly supported wavelets.
    - Symlet (sym4): Near-symmetric wavelet.
    - Coiflet (coif1): Wavelet with additional vanishing moments.
    """)

    st.markdown("### Applications")

    st.write("""
    - Medical image enhancement
    - Satellite image processing
    - Digital photography
    - Image restoration
    - Computer vision
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <strong>WAVELETLAB</strong> · DIGITAL IMAGE PROCESSING

    <br>

    Wavelet-Based Image Restoration System

</div>
""", unsafe_allow_html=True)
