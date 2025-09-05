import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random

# ----- SRCNN Model Architecture -----
def create_srcnn(input_shape=(64, 64, 3)):
    model = Sequential([
        layers.Conv2DTranspose(64, (9,9), strides=4, padding='same', 
                        activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (9,9), padding='same', activation='relu'),
        layers.Conv2D(32, (1,1), padding='same', activation='relu'),
        layers.Conv2D(3, (5,5), padding='same', activation='sigmoid')
    ])
    return model

# Data loading with your specified paths
n = 5000  # Number of images to use
lr_path = "/kaggle/input/sr-data/DATASET/AUG/LR_Aug"
hr_path = "/kaggle/input/sr-data/DATASET/AUG/HR_Aug"

# Load LR images
lr_list = os.listdir(lr_path)[:n]
lr_images = []
for img in lr_list:
    img_path = os.path.join(lr_path, img)
    img_lr = cv2.imread(img_path)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)

# Load HR images
hr_list = os.listdir(hr_path)[:n]
hr_images = []
for img in hr_list:
    img_path = os.path.join(hr_path, img)
    img_hr = cv2.imread(img_path)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)

lr_images = np.array(lr_images) / 255.0
hr_images = np.array(hr_images) / 255.0

#plotting random images for example
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(8,8 ))
plt.subplot(221)
plt.imshow(np.reshape(lr_images[image_number], (64, 64, 3)))
plt.subplot(222)
plt.imshow(np.reshape(hr_images[image_number], (256, 256, 3)))
plt.show()

# Train-test split
lr_train, lr_test, hr_train, hr_test = train_test_split(
    lr_images, hr_images, test_size=0.33, random_state=42
)

# Create TensorFlow datasets
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((lr_train, hr_train)).shuffle(100).batch(batch_size).prefetch(1)
test_dataset = tf.data.Dataset.from_tensor_slices((lr_test, hr_test)).batch(batch_size).prefetch(1)

# ----- Model Initialization -----
srcnn = create_srcnn(lr_train[0].shape)
srcnn.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
srcnn.summary()

# Training parameters
batch_size = 16
epochs = 80

# ----- Training Loop -----
train_losses = []
val_losses = []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_loss = 0
    val_loss = 0
    
    # Training
    for lr_batch, hr_batch in tqdm(train_dataset, desc="Training"):
        with tf.GradientTape() as tape:
            predictions = srcnn(lr_batch, training=True)
            loss = tf.keras.losses.MSE(hr_batch, predictions)
        gradients = tape.gradient(loss, srcnn.trainable_variables)
        srcnn.optimizer.apply_gradients(zip(gradients, srcnn.trainable_variables))
        epoch_loss += tf.reduce_mean(loss)
    
    # Validation
    for lr_val, hr_val in tqdm(test_dataset, desc="Validation"):
        val_predictions = srcnn(lr_val, training=False)
        v_loss = tf.keras.losses.MSE(hr_val, val_predictions)
        val_loss += tf.reduce_mean(v_loss)
    
    # Store metrics
    train_loss = (epoch_loss / len(train_dataset)).numpy()
    valid_loss = (val_loss / len(test_dataset)).numpy()
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")

# Save final model
srcnn.save("srcnn_final.h5")

#----Visualisation and Graphs----#

def evaluate_srcnn(model, test_dataset):
    psnr_values = []
    ssim_values = []
    mse_values = []
    hr_samples = []
    sr_samples = []
    lr_samples = []

    # Disable XLA for prediction
    @tf.function(jit_compile=False)
    def predict_fn(x):
        return model(x, training=False)

    for lr, hr in tqdm(test_dataset, desc="Evaluating SRCNN"):
        # Convert to numpy and back to tensor to avoid graph issues
        lr_np = lr.numpy()
        sr = predict_fn(tf.convert_to_tensor(lr_np))
        
        for i in range(len(hr)):
            hr_img = hr[i].numpy()
            sr_img = sr[i].numpy()
            
            psnr_values.append(psnr(hr_img, sr_img, data_range=1.0))
            ssim_values.append(ssim(hr_img, sr_img, 
                                   data_range=1.0, channel_axis=-1))
            mse_values.append(np.mean((hr_img - sr_img) ** 2))
            
            lr_samples.append(lr_np[i])
            hr_samples.append(hr_img)
            sr_samples.append(sr_img)
    
    return {
        'psnr_values': psnr_values,
        'ssim_values': ssim_values,
        'mse_values': mse_values,
        'lr_samples': np.array(lr_samples),
        'hr_samples': np.array(hr_samples),
        'sr_samples': np.array(sr_samples)
    }

srcnn_metrics = evaluate_srcnn(srcnn, test_dataset)

# Loss Visualization 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, color='blue', marker='o', label='Training Loss')
plt.title('Training Loss Progress')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), val_losses, color='orange', marker='o', label='Validation Loss')
plt.title('Validation Loss Progress')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')

plt.tight_layout()
plt.show()

# Metric Distributions
def plot_srcnn_metrics(srcnn_metrics):
    """Visualize distribution of SRCNN evaluation metrics"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['psnr_values', 'ssim_values', 'mse_values']
    titles = ['PSNR Distribution', 'SSIM Distribution', 'MSE Distribution']
    colors = ['purple', 'green', 'red']
    
    for ax, metric, title, color in zip(axs, metrics, titles, colors):
        ax.hist(srcnn_metrics[metric], bins=30, alpha=0.7, color=color)
        ax.set_title(title)
        ax.set_xlabel(metric.split('_')[0].upper())
        ax.set_ylabel('Frequency')
        
        mean_val = np.mean(srcnn_metrics[metric])
        ax.axvline(mean_val, color='black', linestyle='dashed', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

plot_srcnn_metrics(srcnn_metrics)

# Sample Comparisons 
def plot_srcnn_samples(srcnn_metrics, num_samples=10):
    """Visual comparison with error heatmaps (SRCNN version)"""
    indices = np.random.choice(len(srcnn_metrics['hr_samples']), num_samples)
    
    for idx in indices:
        plt.figure(figsize=(12, 6))
        
        # LR Input
        plt.subplot(241)
        plt.imshow(srcnn_metrics['lr_samples'][idx])
        plt.title('Medium-Resolution Input')
        plt.axis('off')
        
        # HR Ground Truth
        plt.subplot(242)
        plt.imshow(srcnn_metrics['hr_samples'][idx])
        plt.title('High-Resolution Target')
        plt.axis('off')
        
        # SR Output
        plt.subplot(243)
        plt.imshow(srcnn_metrics['sr_samples'][idx])
        metrics_text = f"PSNR: {srcnn_metrics['psnr_values'][idx]:.2f}\n" \
                       f"SSIM: {srcnn_metrics['ssim_values'][idx]:.4f}\n" \
                       f"MSE: {srcnn_metrics['mse_values'][idx]:.5f}"
        plt.title('SRCNN Output\n' + metrics_text)
        plt.axis('off')
        
        # Error Map
        plt.subplot(244)
        error = np.abs(srcnn_metrics['hr_samples'][idx] - srcnn_metrics['sr_samples'][idx])
        plt.imshow(np.mean(error, axis=-1), cmap='inferno')
        plt.title('Pixel Error Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

plot_srcnn_samples(srcnn_metrics)

# Error Analysis
def plot_srcnn_error_analysis(srcnn_metrics):
    """Detailed error characterization for SRCNN"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pixel Error Distribution
    errors = np.concatenate([np.abs(hr - sr) for hr, sr in 
                           zip(srcnn_metrics['hr_samples'], srcnn_metrics['sr_samples'])])
    axs[0].hist(errors.flatten(), bins=50, color='darkorange', density=True)
    axs[0].set_title('Pixel Error Distribution')
    axs[0].set_xlabel('Absolute Error')
    axs[0].set_ylabel('Density')
    
    # Metric Correlation
    axs[1].scatter(srcnn_metrics['psnr_values'], srcnn_metrics['ssim_values'], alpha=0.5)
    axs[1].set_title('PSNR vs SSIM Correlation')
    axs[1].set_xlabel('PSNR')
    axs[1].set_ylabel('SSIM')
    
    
    # Add Trend Line
    psnr_data = np.array(srcnn_metrics['psnr_values'])
    ssim_data = np.array(srcnn_metrics['ssim_values'])
    
    # Fit linear regression through origin
    X = psnr_data.reshape(-1, 1)
    y = ssim_data
    m, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    x_line = np.linspace(0, psnr_data.max(), 100)
    y_line = m[0] * x_line
    
    axs[1].plot(x_line, y_line, color='red', linestyle='--', 
                label=f'Trend: y = {m[0]:.5f}x')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

plot_srcnn_error_analysis(srcnn_metrics)

#----Test with New Dataset ----

#  Load New Dataset 

new_lr_path = "/kaggle/input/test-sr/testing dataset/LR"
new_hr_path = "/kaggle/input/test-sr/testing dataset/HR"

def load_images_from_folder_rgb(folder, resize_shape):
    filenames = sorted(os.listdir(folder))
    images = []
    for file in filenames:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize_shape)
            images.append(img)
    return np.array(images)

# For SRCNN, LR images are 64×64, HR images are 256×256
new_lr_images = load_images_from_folder_rgb(new_lr_path, (64, 64)) / 255.0
new_hr_images = load_images_from_folder_rgb(new_hr_path, (256, 256)) / 255.0

new_test_dataset = tf.data.Dataset.from_tensor_slices((new_lr_images, new_hr_images)).batch(16).prefetch(1)

# Evaluate on New Dataset
new_srcnn_metrics = evaluate_srcnn(srcnn, new_test_dataset)


# Print Side-by-Side Table

orig_psnr = np.mean(srcnn_metrics['psnr_values'])
orig_ssim = np.mean(srcnn_metrics['ssim_values'])
orig_mse  = np.mean(srcnn_metrics['mse_values'])

new_psnr = np.mean(new_srcnn_metrics['psnr_values'])
new_ssim = np.mean(new_srcnn_metrics['ssim_values'])
new_mse  = np.mean(new_srcnn_metrics['mse_values'])

print("\n==================== SRCNN PERFORMANCE COMPARISON ====================")
print(f"{'Metric':<8} | {'Original':>10} | {'New':>10} | {'Diff':>10}")
print("-"*56)

for name, o_val, n_val in zip(["PSNR","SSIM","MSE"], [orig_psnr,orig_ssim,orig_mse], [new_psnr,new_ssim,new_mse]):
    diff = n_val - o_val
    print(f"{name:<8} | {o_val:10.4f} | {n_val:10.4f} | {diff:10.4f}")

#  Plot Bar Charts

plt.figure(figsize=(12,5))
metrics = ["PSNR","SSIM","MSE"]
orig_data = [orig_psnr, orig_ssim, orig_mse]
new_data  = [new_psnr, new_ssim, new_mse]
colors = ["blue","orange"]

for i, (metric, orig_val, new_val) in enumerate(zip(metrics, orig_data, new_data)):
    plt.subplot(1,3,i+1)
    plt.bar(["Original","New"], [orig_val,new_val], color=colors)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)

plt.suptitle("SRCNN Metric Comparison", fontsize=14)
plt.tight_layout()
plt.show()

