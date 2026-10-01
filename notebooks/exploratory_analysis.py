# %% [markdown]
# # SolarCast - Exploratory Data Analysis
# This script analyzes solar magnetogram data to understand features associated with solar flares.

# %%
# Cell 1: Setup and imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set dark style for plots
plt.style.use('dark_background')
os.makedirs('figures', exist_ok=True)

# %%
# Cell 2: Load mock sample data
print("Loading mock dataset...")
np.random.seed(42)

def generate_mock_magnetogram(is_flare=False):
    base = np.random.normal(0, 10, (128, 128))
    if is_flare:
        # Strong bipolar region
        base[50:70, 40:60] = np.random.normal(500, 100, (20, 20))
        base[50:70, 70:90] = np.random.normal(-500, 100, (20, 20))
    return base

n_samples = 100
labels = np.random.binomial(1, 0.3, n_samples)
data = [generate_mock_magnetogram(bool(l)) for l in labels]
df = pd.DataFrame({'id': range(n_samples), 'has_flare': labels})

# %%
# Cell 3: Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='has_flare', palette=['#1f77b4', '#d62728'])
plt.title('Class Distribution (0=Quiet, 1=Flare)')
plt.savefig('figures/class_distribution.png')
plt.close()

# %%
# Cell 4: Example magnetograms
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
quiet_idxs = df[df['has_flare'] == 0].index[:4]
flare_idxs = df[df['has_flare'] == 1].index[:4]

for i, ax in enumerate(axes[0]):
    ax.imshow(data[quiet_idxs[i]], cmap='gray')
    ax.set_title(f'Quiet (ID: {quiet_idxs[i]})')
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    ax.imshow(data[flare_idxs[i]], cmap='gray')
    ax.set_title(f'Flare (ID: {flare_idxs[i]})')
    ax.axis('off')

plt.tight_layout()
plt.savefig('figures/example_magnetograms.png')
plt.close()

# %%
# Cell 5: Image intensity distribution
plt.figure(figsize=(10, 5))
quiet_pixels = np.concatenate([data[i].flatten() for i in quiet_idxs])
flare_pixels = np.concatenate([data[i].flatten() for i in flare_idxs])

plt.hist(quiet_pixels, bins=50, alpha=0.5, label='Quiet', density=True)
plt.hist(flare_pixels, bins=50, alpha=0.5, label='Flare', density=True, color='red')
plt.yscale('log')
plt.title('Pixel Intensity Distribution')
plt.legend()
plt.savefig('figures/intensity_distribution.png')
plt.close()

# %%
# Cell 6: Positive/negative field structure
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sample = data[flare_idxs[0]]

# Positive red, negative blue
colored = np.zeros((*sample.shape, 3))
colored[sample > 100, 0] = 1.0  # Red for positive
colored[sample < -100, 2] = 1.0 # Blue for negative

axes[0].imshow(sample, cmap='gray')
axes[0].set_title('Original Magnetogram')
axes[1].imshow(colored)
axes[1].set_title('Magnetic Polarity (Red=+, Blue=-)')
plt.savefig('figures/polarity_structure.png')
plt.close()

# %%
# Cell 7: Feature distributions
def extract_features(img):
    return {
        'total_flux': np.sum(np.abs(img)),
        'max_field': np.max(np.abs(img)),
        'flux_imbalance': abs(np.sum(img[img>0]) + np.sum(img[img<0])) / (np.sum(np.abs(img)) + 1e-5)
    }

features = pd.DataFrame([extract_features(img) for img in data])
features['has_flare'] = df['has_flare']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=features, x='has_flare', y='total_flux', ax=axes[0])
sns.boxplot(data=features, x='has_flare', y='max_field', ax=axes[1])
sns.boxplot(data=features, x='has_flare', y='flux_imbalance', ax=axes[2])
plt.tight_layout()
plt.savefig('figures/feature_distributions.png')
plt.close()

# %%
# Cell 8: Feature correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(features.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.savefig('figures/correlation_matrix.png')
plt.close()

# %%
# Cell 9: Active region examples
# Showing different synthetic structures
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(generate_mock_magnetogram(True), cmap='magma')
axes[0].set_title('Bipolar Region')
axes[1].imshow(generate_mock_magnetogram(True) * 0.5, cmap='magma')
axes[1].set_title('Weak Region')
axes[2].imshow(generate_mock_magnetogram(False), cmap='magma')
axes[2].set_title('Quiet Sun')
for ax in axes: ax.axis('off')
plt.savefig('figures/active_regions.png')
plt.close()

# %%
# Cell 10: Summary statistics table
summary = features.groupby('has_flare').mean()
print("Average Feature Values by Class:")
print(summary)
summary.to_csv('figures/summary_stats.csv')
