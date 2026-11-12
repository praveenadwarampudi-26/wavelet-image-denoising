import pywt
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load original image
image = cv2.imread("images/input.jpg")

if image is None:
    print("Error: Image not found!") 
else:
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert image to floating-point values
    image_float = image.astype(np.float32)

    # Step 3: Generate Gaussian noise
    mean = 0
    standard_deviation = 25

    noise = np.random.normal(
        mean,
        standard_deviation,
        image.shape
    )

    # Step 4: Add noise to original image
    noisy_image = image_float + noise

    # Keep pixel values between 0 and 255
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)

    # Step 5: Apply Wavelet Denoising
    denoised_channels = []

    for channel in range(3):
        coeffs = pywt.wavedec2(
            noisy_image[:, :, channel],
            'db1',
            level=2
        )

        detail_coeffs = coeffs[-1]
        sigma = np.median(np.abs(detail_coeffs[2])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(noisy_image.shape[0] * noisy_image.shape[1]))

        new_coeffs = [coeffs[0]]

        for details in coeffs[1:]:
            new_details = tuple(
                pywt.threshold(d, threshold, mode='soft')
                for d in details
            )
            new_coeffs.append(new_details)

        reconstructed = pywt.waverec2(new_coeffs, 'db1')
        denoised_channels.append(reconstructed[:noisy_image.shape[0], :noisy_image.shape[1]])

    denoised_image = np.stack(denoised_channels, axis=2)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

    print("Wavelet denoising completed successfully!")
      # Save noisy image
    cv2.imwrite("results/noisy_image.jpg", cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    # Step 5: Display both images
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image)
    plt.title("Image with Gaussian Noise")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Gaussian noise added successfully!")
# Step 6: Display denoised image
plt.figure(figsize=(6, 5))
plt.imshow(denoised_image)
plt.title("Wavelet Denoised Image")
plt.axis("off")
plt.show()

cv2.imwrite("results/denoised_image.jpg",
            cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))

print("Denoised image saved successfully!")