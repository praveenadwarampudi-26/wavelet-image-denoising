
import streamlit as st
import numpy as np
import pywt
import cv2
import time
import io

from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="WaveletLab | Image Denoising",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111a30);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: #10182b;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #a78bfa;
}
.subtitle {
    color: #b6c2d8;
    font-size: 16px;
}
.metric-card {
    background: #192640;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #334568;
    text-align: center;
}
.metric-value {
    font-size: 27px;
    font-weight: bold;
    color: #a78bfa;
}
.metric-label {
    color: #cbd5e1;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- WAVELET DENOISING ----------------
def denoise_channel(channel, wavelet_name, level, threshold_mode):
    channel = channel.astype(np.float64)

    coeffs = pywt.wavedec2(
        channel,
        wavelet_name,
        level=level
    )

    # Estimate noise using the finest diagonal detail coefficients
    diagonal = coeffs[-1][2]
    sigma = np.median(np.abs(diagonal)) / 0.6745

    threshold = sigma * np.sqrt(
        2 * np.log(channel.size)
    )

    processed = [coeffs[0]]

    for detail in coeffs[1:]:
        new_detail = []

        for subband in detail:
            filtered = pywt.threshold(
                subband,
                threshold,
                mode=threshold_mode
            )
            new_detail.append(filtered)

        processed.append(tuple(new_detail))

    restored = pywt.waverec2(
        processed,
        wavelet_name
    )

    restored = restored[:channel.shape[0],
                        :channel.shape[1]]

    return np.clip(restored, 0, 255).astype(np.uint8)


def wavelet_denoise(image, wavelet_name, level, threshold_mode):
    if image.ndim == 2:
        return denoise_channel(
            image, wavelet_name, level, threshold_mode
        )

    channels = []

    for i in range(image.shape[2]):
        channels.append(
            denoise_channel(
                image[:, :, i],
                wavelet_name,
                level,
                threshold_mode
            )
        )

    return np.stack(channels, axis=2)


# ---------------- METRICS ----------------
def calculate_mse(reference, result):
    difference = (
        reference.astype(np.float64)
        - result.astype(np.float64)
    )
    return np.mean(difference ** 2)


def calculate_psnr(reference, result):
    mse = calculate_mse(reference, result)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🔬 WaveletLab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Professional Image Denoising & '
    'Frequency Analysis Platform</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Processing Controls")

page = st.sidebar.radio(
    "Navigation",
    [
        "Image Denoising",
        "Wavelet Theory",
        "Project Information"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Wavelet Image Denoising | DIP Project")

# ---------------- IMAGE DENOISING PAGE ----------------
if page == "Image Denoising":

    st.subheader("Image Processing Workspace")

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:

        original = Image.open(uploaded_file).convert("RGB")
        original = np.array(original)

        st.success("Image uploaded successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(original, use_container_width=True)

        with col2:
            st.markdown("### Processing Settings")

            add_noise = st.checkbox(
                "Simulate Gaussian noise",
                value=True
            )

            noise_level = st.slider(
                "Noise intensity",
                min_value=1,
                max_value=100,
                value=25,
                disabled=not add_noise
            )

            wavelet_name = st.selectbox(
                "Select Wavelet",
                ["haar", "db2", "db4", "sym4", "coif1"]
            )

            max_level = pywt.dwt_max_level(
                min(original.shape[:2]),
                pywt.Wavelet(wavelet_name).dec_len
            )

            max_level = max(1, min(max_level, 5))

            level = st.slider(
                "Decomposition Level",
                min_value=1,
                max_value=max_level,
                value=1
            )

            threshold_mode = st.selectbox(
                "Threshold Method",
                ["soft", "hard"]
            )

        st.divider()

        if st.button(
            "🚀 Start Wavelet Denoising",
            type="primary",
            use_container_width=True
        ):

            # Prepare noisy input
            if add_noise:
                rng = np.random.default_rng()
                noise = rng.normal(
                    0, noise_level, original.shape
                )

                noisy = np.clip(
                    original.astype(np.float64) + noise,
                    0, 255
                ).astype(np.uint8)
            else:
                noisy = original.copy()

            # Process image
            start_time = time.perf_counter()

            denoised = wavelet_denoise(
                noisy,
                wavelet_name,
                level,
                threshold_mode
            )

            processing_time = time.perf_counter() - start_time

            # Metrics against original reference
            mse_noisy = calculate_mse(original, noisy)
            mse_denoised = calculate_mse(original, denoised)

            psnr_noisy = calculate_psnr(original, noisy)
            psnr_denoised = calculate_psnr(original, denoised)

            st.session_state["noisy"] = noisy
            st.session_state["denoised"] = denoised
            st.session_state["original"] = original
            st.session_state["processing_time"] = processing_time
            st.session_state["mse_noisy"] = mse_noisy
            st.session_state["mse_denoised"] = mse_denoised
            st.session_state["psnr_noisy"] = psnr_noisy
            st.session_state["psnr_denoised"] = psnr_denoised

            st.success("Wavelet denoising completed successfully!")

        # Display results
        if "denoised" in st.session_state:

            st.divider()
            st.subheader("Image Comparison")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Original**")
                st.image(
                    st.session_state["original"],
                    use_container_width=True
                )

            with c2:
                st.markdown("**Noisy Image**")
                st.image(
                    st.session_state["noisy"],
                    use_container_width=True
                )

            with c3:
                st.markdown("**Denoised Image**")
                st.image(
                    st.session_state["denoised"],
                    use_container_width=True
                )

            st.divider()
            st.subheader("📊 Performance Analysis")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "PSNR After Denoising",
                    f"{st.session_state['psnr_denoised']:.2f} dB",
                    delta=f"{st.session_state['psnr_denoised'] - st.session_state['psnr_noisy']:.2f} dB"
                )

            with m2:
                st.metric(
                    "MSE After Denoising",
                    f"{st.session_state['mse_denoised']:.2f}",
                    delta=f"{st.session_state['mse_denoised'] - st.session_state['mse_noisy']:.2f}",
                    delta_color="inverse"
                )

            with m3:
                st.metric(
                    "Processing Time",
                    f"{st.session_state['processing_time']:.4f} sec"
                )

            st.caption(
                "Metrics are calculated against the original uploaded "
                "image, which acts as the clean reference in this "
                "noise-simulation experiment."
            )

            st.divider()
            st.subheader("📈 Wavelet Analysis")

            gray = cv2.cvtColor(
                st.session_state["noisy"],
                cv2.COLOR_RGB2GRAY
            )

            coefficients = pywt.wavedec2(
                gray,
                wavelet_name,
                level=level
            )

            approximation = coefficients[0]

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.imshow(approximation, cmap="gray")
            ax.set_title("Wavelet Approximation Coefficients")
            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

            st.divider()
            st.subheader("Download Your Results")

            output = io.BytesIO()

            Image.fromarray(
                st.session_state["denoised"]
            ).save(output, format="PNG")

            st.download_button(
                "⬇️ Download Denoised Image",
                data=output.getvalue(),
                file_name="wavelet_denoised.png",
                mime="image/png",
                use_container_width=True
            )

    else:
        st.info(
            "👆 Upload an image to begin your wavelet denoising experiment."
        )

        st.markdown("""
        ### Platform Features

        - Image upload and preprocessing
        - Gaussian noise simulation
        - Discrete Wavelet Transform
        - Soft and hard thresholding
        - Image restoration
        - PSNR and MSE evaluation
        - Download restored images
        """)

# ---------------- THEORY PAGE ----------------
elif page == "Wavelet Theory":

    st.title("Wavelet Transform Theory")

    st.markdown("""
    ### 1. What is Wavelet Transform?

    Wavelet Transform is a mathematical technique used to
    represent an image at different scales and resolutions.

    ### 2. Image Decomposition

    The 2D Discrete Wavelet Transform divides an image into
    four subbands:

    - **LL:** Approximation coefficients
    - **LH:** Horizontal detail coefficients
    - **HL:** Vertical detail coefficients
    - **HH:** Diagonal detail coefficients

    ### 3. Denoising Process

    1. Decompose the noisy image using DWT.
    2. Estimate the noise level.
    3. Apply thresholding to detail coefficients.
    4. Reconstruct the image using inverse DWT.

    ### 4. Mean Squared Error (MSE)

    MSE measures the average squared difference between
    the original and restored images.

    MSE = (1/MN) Σ [I(i,j) - K(i,j)]²

    ### 5. Peak Signal-to-Noise Ratio (PSNR)

    PSNR measures the reconstruction quality in decibels.

    PSNR = 10 log10(MAX² / MSE)
    """)

# ---------------- PROJECT INFORMATION ----------------
elif page == "Project Information":

    st.title("About This Project")

    st.markdown("""
    ### Wavelet Image Denoising Using Python

    **Domain:** Digital Image Processing

    **Objective:** To reduce unwanted noise from digital
    images while preserving important image details.

    **Technologies:**
    - Python
    - OpenCV
    - NumPy
    - PyWavelets
    - Streamlit
    - Matplotlib

    **Applications:**
    - Medical image processing
    - Satellite image enhancement
    - Photography
    - Computer vision
    - Scientific imaging
    """)

st.divider()

st.caption(
    "WaveletLab | Digital Image Processing Academic Project"
)
