import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import (Conv2D, Lambda, LeakyReLU, Dense, Input, add, concatenate, Flatten, BatchNormalization, PReLU, UpSampling2D)
from keras.applications.vgg19 import VGG19, preprocess_input

# --- Helper Function to Convert BGR -> RGB for Display ---
def bgr_to_rgb(image):
    """Convert a BGR image (H,W,3) to RGB for display purposes."""
    return image[..., ::-1]

# --- Learning Rate Schedule ---
def lr_schedule(epoch):
    if epoch < 10:
        return 1e-5  # Warmup phase
    else:
        return 1e-4 * (0.95 ** (epoch // 10))

# Initialize optimizers 
gen_optimizer = tf.keras.optimizers.Adam(lr_schedule(0))
disc_optimizer = tf.keras.optimizers.Adam(1e-5)  # Generator LR=1e-4, Discriminator LR=1e-5

# SRGAN Model Components

# Residual Block for SRGAN
def residual_block(input_tensor, filters=64):
    x = Conv2D(filters, (3,3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    return add([input_tensor, x])

# Upscale Block with sub-pixel convolution (using UpSampling2D)
def upscale_block(input_tensor, filters=256):
    x = UpSampling2D(size=2)(input_tensor)
    x = Conv2D(filters, (3,3), padding='same')(x)
    return PReLU(shared_axes=[1,2])(x)

# SRGAN Generator
def create_gen(gen_ip, num_res_block=16):
    x = Conv2D(64, (3,3), padding='same')(gen_ip)
    x = PReLU(shared_axes=[1,2])(x)
    temp = x
    for _ in range(num_res_block):
        x = residual_block(x)
    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, temp])
    x = upscale_block(x, 256)
    x = upscale_block(x, 256)
    x = Conv2D(3, (3,3), padding='same', activation='sigmoid', dtype='float32')(x)
    return Model(gen_ip, x)

# SRGAN Discriminator
def create_disc(disc_ip):
    df = 64
    x = Conv2D(df, (3,3), padding='same')(disc_ip)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*2, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*2, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*4, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*4, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*8, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(df*8, (3,3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(df*16)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(disc_ip, x)

# VGG19 Feature Extractor: 
def build_vgg(hr_shape):
    inp = Input(shape=hr_shape, dtype='float32')
    preprocessed = Lambda(lambda x: preprocess_input(x * 255.0), output_shape=lambda s: s)(inp)
    base_vgg = VGG19(weights="imagenet", include_top=False, input_tensor=preprocessed)
    features = base_vgg.get_layer("block3_conv3").output
    reduced_features = Conv2D(64, (1,1), padding='same', activation='linear')(features)
    return Model(inputs=inp, outputs=reduced_features)

# Combined model for perceptual loss
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

# Data Loading
n = 5000  # Number of images to use
lr_path = "/kaggle/input/sr-dataset/DATASET/AUG/LR_Aug"
hr_path = "/kaggle/input/sr-dataset/DATASET/AUG/HR_Aug"

lr_list = os.listdir(lr_path)[:n]
hr_list = os.listdir(hr_path)[:n]

lr_images = []
hr_images = []

# Load images and keep them in BGR (for VGG)
for img in lr_list:
    img_path = os.path.join(lr_path, img)
    img_lr = cv2.imread(img_path)  # BGR format
    lr_images.append(img_lr)

for img in hr_list:
    img_path = os.path.join(hr_path, img)
    img_hr = cv2.imread(img_path)  # BGR format
    hr_images.append(img_hr)

lr_images = np.array(lr_images, dtype=np.float32) / 255.0
hr_images = np.array(hr_images, dtype=np.float32) / 255.0

print(f"LR shape: {lr_images[0].shape}, HR shape: {hr_images[0].shape}")
print(f"LR range: [{lr_images.min()}, {lr_images.max()}]")
print(f"HR range: [{hr_images.min()}, {hr_images.max()}]")

# Visualizing Random Images
random_idx = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(8,8))
plt.subplot(2, 2, 1)
plt.imshow(bgr_to_rgb(lr_images[random_idx]))
plt.title(" LR Sample ")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(bgr_to_rgb(hr_images[random_idx]))
plt.title(" HR Sample ")
plt.axis("off")
plt.show()

# Train-test split
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=42)
hr_shape = (256,256,3)
lr_shape = (64,64,3)

# Model Initialization
generator = create_gen(Input(lr_shape))
generator.summary()

discriminator = create_disc(Input(hr_shape))
discriminator.compile(loss="binary_crossentropy", optimizer=disc_optimizer, metrics=['accuracy'])
discriminator.summary()

vgg = build_vgg(hr_shape)
print(vgg.summary())
vgg.trainable = False

# Training Parameters
batch_size = 6
epochs = 80 # Adjust as needed
train_dataset = tf.data.Dataset.from_tensor_slices((lr_train, hr_train)).batch(batch_size)

# Loss tracking lists
gen_losses = []
disc_losses = []

# Training Loop 
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_gen_loss = 0.0
    epoch_disc_loss = 0.0
    num_batches = 0
    
    for lr_batch, hr_batch in tqdm(train_dataset, desc="Training Epoch"):
        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            fake_hr = generator(lr_batch, training=True)
            real_output = discriminator(hr_batch, training=True)
            fake_output = discriminator(fake_hr, training=True)
            
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_output),
                logits=real_output - tf.reduce_mean(fake_output)
            ))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_output),
                logits=fake_output - tf.reduce_mean(real_output)
            ))
            disc_loss = 0.5 * (real_loss + fake_loss)
        
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_weights))
        
        # Train Generator
        with tf.GradientTape() as gen_tape:
            fake_hr = generator(lr_batch, training=True)
            fake_output = discriminator(fake_hr, training=False)
            real_output = discriminator(hr_batch, training=False)
            
            gen_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_output),
                logits=fake_output - tf.reduce_mean(real_output)
            ))
            
            gen_features = vgg(fake_hr)
            real_features = vgg(hr_batch)
            gen_content_loss = tf.reduce_mean(tf.square(gen_features - real_features))
            
            # Increase adversarial weight to 1e-2 for generator priority
            gen_total_loss = gen_content_loss + 1e-2 * gen_adv_loss
            
        gen_grads = gen_tape.gradient(gen_total_loss, generator.trainable_weights)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))
        
        epoch_gen_loss += gen_total_loss.numpy()
        epoch_disc_loss += disc_loss.numpy()
        num_batches += 1
    
    avg_gen_loss = epoch_gen_loss / num_batches
    avg_disc_loss = epoch_disc_loss / num_batches
    print(f"Epoch {epoch+1}: Generator Loss = {avg_gen_loss:.4f}, Discriminator Loss = {avg_disc_loss:.4f}")
    
    gen_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)

# Save final generator
generator.save("srgan_generator_final.h5")

generator.save("srgan_generator_final.h5")

# Evaluation Metrics
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

gan_metrics = evaluate_gan(generator, lr_test, hr_test)
print("Mean PSNR:", np.mean(gan_metrics['psnr']))
print("Mean SSIM:", np.mean(gan_metrics['ssim']))
print("Mean MSE: ", np.mean(gan_metrics['mse']))

# Loss Visualization 
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), gen_losses, color='blue', marker='o', label='Generator Loss')
plt.title('Generator Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs+1), disc_losses, color='orange', marker='o', label='Discriminator Loss')
plt.title('Discriminator Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Sample Comparisons
def plot_gan_samples(gan_metrics, num_samples=3):
    indices = np.random.choice(len(gan_metrics['lr_samples']), num_samples)
    for idx in indices:
        plt.figure(figsize=(12,6))
        # LR Input (BGR -> RGB for display)
        plt.subplot(241)
        lr_bgr = gan_metrics['lr_samples'][idx]
        plt.imshow(bgr_to_rgb(lr_bgr))
        plt.title('LR Input')
        plt.axis('off')
        
        # HR Target (BGR -> RGB for display)
        plt.subplot(242)
        hr_bgr = gan_metrics['hr_samples'][idx]
        plt.imshow(bgr_to_rgb(hr_bgr))
        plt.title('HR Target')
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

# Metric Distributions
def plot_gan_metrics(gan_metrics):
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    metrics = ['psnr', 'ssim', 'mse']
    titles = ['PSNR Distribution', 'SSIM Distribution', 'MSE Distribution']
    colors = ['purple', 'green', 'red']
    for ax, metric, title, color in zip(axs, metrics, titles, colors):
        ax.hist(gan_metrics[metric], bins=30, alpha=0.7, color=color)
        ax.set_title(title)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frequency')
        mean_val = np.mean(gan_metrics[metric])
        ax.axvline(mean_val, color='black', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend()
    plt.tight_layout()
    plt.show()

plot_gan_metrics(gan_metrics)

# Error Analysis
def plot_gan_error_analysis(gan_metrics):
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    errors = np.concatenate([np.abs(hr - sr) for hr, sr in zip(gan_metrics['hr_samples'], gan_metrics['sr_samples'])])
    axs[0].hist(errors.flatten(), bins=50, color='darkorange', density=True)
    axs[0].set_title('Pixel Error Distribution')
    axs[0].set_xlabel('Absolute Error')
    axs[0].set_ylabel('Density')
    
    axs[1].scatter(gan_metrics['psnr'], gan_metrics['ssim'], alpha=0.5)
    axs[1].set_title('PSNR vs SSIM Correlation')
    axs[1].set_xlabel('PSNR')
    axs[1].set_ylabel('SSIM')
    
    psnr_data = np.array(gan_metrics['psnr'])
    ssim_data = np.array(gan_metrics['ssim'])
    X = psnr_data.reshape(-1, 1)
    y = ssim_data
    m, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    x_line = np.linspace(0, psnr_data.max(), 100)
    y_line = m[0] * x_line
    axs[1].plot(x_line, y_line, color='red', linestyle='--', label=f'Trend: y = {m[0]:.5f}x')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

plot_gan_error_analysis(gan_metrics)

#----Test with New Dataset ----

#  Load New Dataset (SRGAN)

new_lr_dir = "/kaggle/input/test-sr/testing dataset/LR"
new_hr_dir = "/kaggle/input/test-sr/testing dataset/HR"

def load_images_rgb(folder, resize_shape):
    filenames = sorted(os.listdir(folder))
    images = []
    for file in filenames:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, resize_shape)
            images.append(img)
    return np.array(images, dtype=np.float32)/255.0

# For SRGAN, LR=64×64, HR=256×256
new_lr_images = load_images_rgb(new_lr_dir, (64,64))
new_hr_images = load_images_rgb(new_hr_dir, (256,256))


# Evaluate on New Dataset

new_gan_metrics = evaluate_gan(generator, new_lr_images, new_hr_images)


# Print Side-by-Side Table
orig_psnr = np.mean(gan_metrics['psnr'])
orig_ssim = np.mean(gan_metrics['ssim'])
orig_mse  = np.mean(gan_metrics['mse'])

new_psnr = np.mean(new_gan_metrics['psnr'])
new_ssim = np.mean(new_gan_metrics['ssim'])
new_mse  = np.mean(new_gan_metrics['mse'])

print("\n==================== SRGAN PERFORMANCE COMPARISON ====================")
print(f"{'Metric':<8} | {'Original':>10} | {'New':>10} | {'Diff':>10}")
print("-"*56)

for name, o_val, n_val in zip(["PSNR","SSIM","MSE"], [orig_psnr,orig_ssim,orig_mse], [new_psnr,new_ssim,new_mse]):
    diff = n_val - o_val
    print(f"{name:<8} | {o_val:10.4f} | {n_val:10.4f} | {diff:10.4f}")

# Plot Bar Charts

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

plt.suptitle("SRGAN Metric Comparison", fontsize=14)
plt.tight_layout()
plt.show()