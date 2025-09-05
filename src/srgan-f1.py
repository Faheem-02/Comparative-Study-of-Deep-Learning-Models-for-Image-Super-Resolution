import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Load trained SRGAN generator model
model_path = "/kaggle/input/srgan/other/default/1/srgan_generator_final.h5"  # Update with actual path
generator = load_model(model_path, compile=False)

# Function to load images in RGB format
def load_images_rgb(folder, resize_shape):
    filenames = sorted(os.listdir(folder))
    images = []
    for file in filenames:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, resize_shape)
            images.append(img.astype(np.float32) / 255.0)  # Normalize to [0,1]
    return np.array(images)

# Assuming you have a folder structure for the test dataset
lr_test_dir = "/kaggle/input/sr-dataset/DATASET/AUG/LR_Aug"
hr_test_dir = "/kaggle/input/sr-dataset/DATASET/AUG/HR_Aug"

# Load your test data (e.g., 64x64 LR images and 256x256 HR images)
lr_test = load_images_rgb(lr_test_dir, (64, 64))
hr_test = load_images_rgb(hr_test_dir, (256, 256))



def evaluate_gan(generator, lr_test, hr_test, batch_size=8):
    psnr_values = []
    ssim_values = []
    mse_values = []
    lr_samples = []
    hr_samples = []
    sr_samples = []
    
    for i in tqdm(range(0, len(lr_test), batch_size), desc="Generating SR images"):
        lr_batch = lr_test[i:i+batch_size]
        hr_batch = hr_test[i:i+batch_size]
        
        sr_batch = generator.predict(lr_batch, verbose=0)
        sr_batch = np.clip(sr_batch, 0.0, 1.0)
        
        for j in range(len(hr_batch)):
            psnr_val = psnr(hr_batch[j], sr_batch[j], data_range=1.0)
            ssim_val = ssim(hr_batch[j], sr_batch[j], data_range=1.0, channel_axis=-1)
            mse_val = np.mean((hr_batch[j] - sr_batch[j]) ** 2)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            mse_values.append(mse_val)
        
        lr_samples.extend(lr_batch)
        hr_samples.extend(hr_batch)
        sr_samples.extend(sr_batch)
    
    return {
        'psnr': psnr_values,
        'ssim': ssim_values,
        'mse': mse_values,
        'lr_samples': np.array(lr_samples),
        'hr_samples': np.array(hr_samples),
        'sr_samples': np.array(sr_samples)
    }
    # Now you can evaluate your GAN
gan_metrics = evaluate_gan(generator, lr_test, hr_test)
# Evaluation Metrics

gan_metrics = evaluate_gan(generator, lr_test, hr_test)


print("PSNR:", np.max(gan_metrics['psnr']))


print("SSIM:", np.max(gan_metrics['ssim']))


print("MSE:", np.min(gan_metrics['mse'])) 

import numpy as np
import matplotlib.pyplot as plt

# Helper function to convert BGR to RGB
def bgr_to_rgb(bgr_image):
    return bgr_image[..., ::-1]

# Sample Comparisons with PSNR filter
def plot_gan_samples(gan_metrics, min_psnr=20, num_samples=250):
    # Filter indices where PSNR is greater than the threshold
    valid_indices = np.where(np.array(gan_metrics['psnr']) >= min_psnr)[0]
    
    # Randomly select from the filtered indices
    indices = np.random.choice(valid_indices, num_samples)
    
    for idx in indices:
        plt.figure(figsize=(12,6))
        
        # LR Input (BGR -> RGB for display)
        plt.subplot(241)
        lr_bgr = gan_metrics['lr_samples'][idx]
        plt.imshow(bgr_to_rgb(lr_bgr))
        plt.title('Medium-Resolution Input')
        plt.axis('off')
        
        # HR Target (BGR -> RGB for display)
        plt.subplot(242)
        hr_bgr = gan_metrics['hr_samples'][idx]
        plt.imshow(bgr_to_rgb(hr_bgr))
        plt.title('High-Resolution Target')
        plt.axis('off')
        
        # SR Output (BGR -> RGB for display) 
        plt.subplot(243)
        sr_bgr = gan_metrics['sr_samples'][idx]
        metrics_text = f"PSNR: {gan_metrics['psnr'][idx]:.2f}\nSSIM: {gan_metrics['ssim'][idx]:.4f}\nMSE: {gan_metrics['mse'][idx]:.5f}"
        plt.imshow(bgr_to_rgb(sr_bgr))
        plt.title('SRGAN Output\n' + metrics_text)
        plt.axis('off')
        
        # Error Map
        plt.subplot(244)
        error = np.abs(hr_bgr - sr_bgr)
        plt.imshow(np.mean(error, axis=-1), cmap='inferno', vmin=0, vmax=0.07)
        plt.title('Pixel Error Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


plot_gan_samples(gan_metrics)

# Load New Dataset
new_lr_dir = "/kaggle/input/test-sr/testing dataset/LR"
new_hr_dir = "/kaggle/input/test-sr/testing dataset/HR"
new_lr_images = load_images_rgb(new_lr_dir, (64,64))
new_hr_images = load_images_rgb(new_hr_dir, (256,256))

import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_gan(generator, lr_test, hr_test, batch_size=8):
    psnr_values = []
    ssim_values = []
    mse_values = []
    lr_samples = []
    hr_samples = []
    sr_samples = []
    
    for i in tqdm(range(0, len(lr_test), batch_size), desc="Generating SR images"):
        lr_batch = lr_test[i:i+batch_size]
        hr_batch = hr_test[i:i+batch_size]
        
        sr_batch = generator.predict(lr_batch, verbose=0)
        sr_batch = np.clip(sr_batch, 0.0, 1.0)
        
        for j in range(len(hr_batch)):
            psnr_val = psnr(hr_batch[j], sr_batch[j], data_range=1.0)
            ssim_val = ssim(hr_batch[j], sr_batch[j], data_range=1.0, channel_axis=-1)
            mse_val = np.mean((hr_batch[j] - sr_batch[j]) ** 2)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            mse_values.append(mse_val)
        
        lr_samples.extend(lr_batch)
        hr_samples.extend(hr_batch)
        sr_samples.extend(sr_batch)
    
    return {
        'psnr': psnr_values,
        'ssim': ssim_values,
        'mse': mse_values,
        'lr_samples': np.array(lr_samples),
        'hr_samples': np.array(hr_samples),
        'sr_samples': np.array(sr_samples)
    }

# Evaluate on original dataset
gan_metrics = evaluate_gan(generator, lr_test, hr_test)
orig_psnr_max = np.max(gan_metrics['psnr'])
orig_ssim_max = np.max(gan_metrics['ssim'])
orig_mse_min = np.min(gan_metrics['mse'])

# Evaluate on new dataset
new_gan_metrics = evaluate_gan(generator, new_lr_images, new_hr_images)
new_psnr_max = np.max(new_gan_metrics['psnr'])
new_ssim_max = np.max(new_gan_metrics['ssim'])
new_mse_min = np.min(new_gan_metrics['mse'])

# Print side-by-side performance comparison
print("\n==================== SRGAN PERFORMANCE COMPARISON ====================")
print(f"{'Metric':<8} | {'Original':>14} | {'New':>14} | {'Diff':>14}")
print("-" * 60)
for name, o_best, n_best in zip(
    ["PSNR", "SSIM", "MSE"], 
    [orig_psnr_max, orig_ssim_max, orig_mse_min], 
    [new_psnr_max, new_ssim_max, new_mse_min]
):
    diff = n_best - o_best
    print(f"{name:<8} | {o_best:14.4f} | {n_best:14.4f} | {diff:14.4f}")

# Plotting the Bar Charts for Comparison
plt.figure(figsize=(12, 5))

# Data for plotting
metrics = ["PSNR", "SSIM", "MSE"]
orig_data = [orig_psnr_max, orig_ssim_max, orig_mse_min]
new_data = [new_psnr_max, new_ssim_max, new_mse_min]
colors = ["blue", "orange"]

# Loop through the metrics and plot bar charts
for i, (metric, orig_val, new_val) in enumerate(zip(metrics, orig_data, new_data)):
    plt.subplot(1, 3, i + 1)
    plt.bar(["Original", "New"], [orig_val, new_val], color=colors)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)

plt.suptitle("SRGAN Metric Comparison", fontsize=14)
plt.tight_layout()
plt.show()

