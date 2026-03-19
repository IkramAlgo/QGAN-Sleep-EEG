# qgan/plot_ibm_results.py
# DYNAMIC PLOT CODE — No emojis, no font warnings
# Run: python -m qgan.plot_ibm_results

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==================== CONFIGURATION ====================
# Change this to point to whatever JSON file you want to plot
JSON_FILENAME = "results_ibm_improved_3epochs.json"  # Change to "results_ibm_improved_50epochs.json" later

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#59a14f', '#76b7b2', '#edc949']

# ==================== LOAD DATA ====================
print(f"\n Loading results from: {JSON_FILENAME}")
try:
    with open(JSON_FILENAME, "r") as f:
        results = json.load(f)
    print(f" Loaded {len(results)} feature configurations")
except FileNotFoundError:
    print(f" File not found: {JSON_FILENAME}")
    print("   Please check the filename or run training first.")
    exit(1)

# ==================== EXTRACT DATA ====================
features = [r["n_features"] for r in results]
qubits = [r["n_qubits"] for r in results]
accuracy = [r["clf"]["Accuracy"] for r in results]
precision = [r["clf"]["Precision"] for r in results]
sensitivity = [r["clf"]["Sensitivity"] for r in results]
specificity = [r["clf"]["Specificity"] for r in results]
f1 = [r["clf"]["F1"] for r in results]

# Get MAE data (handle cases where history might be empty)
std_mae = []
mean_mae = []
for r in results:
    if r["history"] and r["history"].get("std_MAE") and len(r["history"]["std_MAE"]) > 0:
        std_mae.append(r["history"]["std_MAE"][-1])  # Last epoch
        mean_mae.append(r["history"]["mean_MAE"][-1])
    else:
        std_mae.append(0.0)
        mean_mae.append(0.0)

# Get epochs trained
epochs_trained = []
for r in results:
    if r["history"] and r["history"].get("gen_loss"):
        epochs_trained.append(len(r["history"]["gen_loss"]))
    else:
        epochs_trained.append(r["epochs"])

# ==================== FIGURE 1: Bar Chart Comparison ====================
plt.figure(figsize=(14, 7))
x = np.arange(len(features))
width = 0.15

bars1 = plt.bar(x - 2*width, accuracy, width, label='Accuracy', color=COLORS[0], alpha=0.9)
bars2 = plt.bar(x - width, precision, width, label='Precision', color=COLORS[1], alpha=0.9)
bars3 = plt.bar(x, sensitivity, width, label='Sensitivity', color=COLORS[2], alpha=0.9)
bars4 = plt.bar(x + width, specificity, width, label='Specificity', color=COLORS[3], alpha=0.9)
bars5 = plt.bar(x + 2*width, f1, width, label='F1 Score', color=COLORS[4], alpha=0.9)

plt.xlabel('Number of Features', fontsize=14, fontweight='bold')
plt.ylabel('Score', fontsize=14, fontweight='bold')
plt.title(f'IBM QPU Simulation Results ({epochs_trained[0]} Epochs)', fontsize=16, fontweight='bold')
plt.xticks(x, [f'{f} Features\n({q} Qubits)' for f, q in zip(features, qubits)], fontsize=12)
plt.ylim(0, 1.1)
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig('figures/ibm_results_complete.png', dpi=150, bbox_inches='tight')
print(" Saved: figures/ibm_results_complete.png")

# ==================== FIGURE 2: MAE Comparison ====================
plt.figure(figsize=(12, 6))
x = np.arange(len(features))
width = 0.35

bars1 = plt.bar(x - width/2, mean_mae, width, label='Mean MAE', color=COLORS[0], alpha=0.8)
bars2 = plt.bar(x + width/2, std_mae, width, label='Std MAE', color=COLORS[1], alpha=0.8)

plt.xlabel('Number of Features', fontsize=14, fontweight='bold')
plt.ylabel('MAE (lower is better)', fontsize=14, fontweight='bold')
plt.title(f'Distribution Matching Quality ({epochs_trained[0]} Epochs)', fontsize=16, fontweight='bold')
plt.xticks(x, [f'{f} Features\n({q} Qubits)' for f, q in zip(features, qubits)], fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/ibm_mae_comparison.png', dpi=150, bbox_inches='tight')
print(" Saved: figures/ibm_mae_comparison.png")

# ==================== FIGURE 3: Training Curves (All Features with History) ====================
features_with_history = [i for i, r in enumerate(results) if r["history"] and r["history"].get("gen_loss")]

if features_with_history:
    n_plots = len(features_with_history)
    fig, axes = plt.subplots(n_plots, 3, figsize=(18, 5*n_plots))
    
    # Handle single row case
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    for idx, plot_idx in enumerate(features_with_history):
        r = results[plot_idx]
        feat = r["n_features"]
        epochs = range(1, len(r["history"]["gen_loss"]) + 1)
        
        # Generator Loss
        axes[idx, 0].plot(epochs, r["history"]["gen_loss"], 'o-', color=COLORS[0], linewidth=2, markersize=6)
        axes[idx, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero line')
        axes[idx, 0].set_xlabel('Epoch', fontsize=12)
        axes[idx, 0].set_ylabel('Generator Loss', fontsize=12)
        axes[idx, 0].set_title(f'{feat} Features: Generator Loss', fontsize=14, fontweight='bold')
        axes[idx, 0].grid(alpha=0.3)
        axes[idx, 0].legend()
        
        # Discriminator Loss
        axes[idx, 1].plot(epochs, r["history"]["disc_loss"], 'o-', color=COLORS[1], linewidth=2, markersize=6)
        axes[idx, 1].set_xlabel('Epoch', fontsize=12)
        axes[idx, 1].set_ylabel('Discriminator Loss', fontsize=12)
        axes[idx, 1].set_title(f'{feat} Features: Discriminator Loss', fontsize=14, fontweight='bold')
        axes[idx, 1].grid(alpha=0.3)
        
        # MINE Loss
        if r["history"].get("mine_loss"):
            axes[idx, 2].plot(epochs, r["history"]["mine_loss"], 'o-', color=COLORS[2], linewidth=2, markersize=6)
            axes[idx, 2].set_xlabel('Epoch', fontsize=12)
            axes[idx, 2].set_ylabel('MINE Loss', fontsize=12)
            axes[idx, 2].set_title(f'{feat} Features: MINE Loss', fontsize=14, fontweight='bold')
            axes[idx, 2].grid(alpha=0.3)
            
            # Add healthy range indicator
            axes[idx, 2].axhspan(0.02, 0.10, alpha=0.2, color='green', label='Healthy range')
            axes[idx, 2].legend()
    
    plt.suptitle(f'Training Dynamics ({epochs_trained[0]} Epochs)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ibm_training_curves.png', dpi=150, bbox_inches='tight')
    print(" Saved: figures/ibm_training_curves.png")

# ==================== FIGURE 4: MAE Over Training ====================
features_with_mae = [i for i, r in enumerate(results) if r["history"] and r["history"].get("mae_epochs")]

if features_with_mae:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx in features_with_mae:
        r = results[idx]
        feat = r["n_features"]
        
        axes[0].plot(r["history"]["mae_epochs"], r["history"]["mean_MAE"], 
                    'o-', linewidth=2, markersize=8, label=f'{feat} Features', color=COLORS[idx])
        axes[1].plot(r["history"]["mae_epochs"], r["history"]["std_MAE"], 
                    'o-', linewidth=2, markersize=8, label=f'{feat} Features', color=COLORS[idx])
    
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Mean MAE', fontsize=14)
    axes[0].set_title('Mean Matching Over Training', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Std MAE', fontsize=14)
    axes[1].set_title('Variance Matching Over Training', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f'MAE Convergence ({epochs_trained[0]} Epochs)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ibm_mae_convergence.png', dpi=150, bbox_inches='tight')
    print(" Saved: figures/ibm_mae_convergence.png")

# ==================== FIGURE 5: Collapse Analysis ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Generator Loss at Final Epoch
ax = axes[0, 0]
final_gen_loss = [r["history"]["gen_loss"][-1] if r["history"] and r["history"].get("gen_loss") else 0 for r in results]
bars = ax.bar([f'{f} Feat' for f in features], final_gen_loss, color=COLORS[:len(features)])
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax.set_xlabel('Feature Configuration', fontsize=12)
ax.set_ylabel('Generator Loss (Final Epoch)', fontsize=12)
ax.set_title('Generator Loss Analysis', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Color bars based on collapse risk
for i, bar in enumerate(bars):
    if final_gen_loss[i] < 0:
        bar.set_color('red')
        bar.set_alpha(0.7)
    elif i == 1 and specificity[i] > 0.9 and f1[i] < 0.1:
        bar.set_color('orange')
        bar.set_alpha(0.7)

# Plot 2: Specificity vs F1 Scatter
ax = axes[0, 1]
for i, (spec, f1_val, feat) in enumerate(zip(specificity, f1, features)):
    if i == 0:
        ax.scatter(spec, f1_val, s=200, c=COLORS[i], marker='o', label=f'{feat} Feat', edgecolors='black', linewidth=2)
    else:
        ax.scatter(spec, f1_val, s=200, c=COLORS[i], marker='o', label=f'{feat} Feat', edgecolors='black', linewidth=2)
    
    # Add annotation
    ax.annotate(f'{feat}F', (spec, f1_val), xytext=(5, 5), textcoords='offset points', fontsize=11)

ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Specificity', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Specificity vs F1 Score', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Plot 3: MINE Loss Comparison
ax = axes[1, 0]
for i, r in enumerate(results):
    if r["history"] and r["history"].get("mine_loss"):
        epochs = range(1, len(r["history"]["mine_loss"]) + 1)
        ax.plot(epochs, r["history"]["mine_loss"], 'o-', linewidth=2, markersize=8, 
                label=f'{features[i]} Features', color=COLORS[i])

ax.axhspan(0.02, 0.10, alpha=0.2, color='green', label='Healthy range')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MINE Loss', fontsize=12)
ax.set_title('MINE Loss Evolution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Summary Text
ax = axes[1, 1]
ax.axis('off')

# Create summary text (no emojis)
summary_text = "COLLAPSE ANALYSIS\n" + "="*30 + "\n\n"
for i, feat in enumerate(features):
    summary_text += f"{feat} Features:\n"
    summary_text += f"  Accuracy: {accuracy[i]:.3f}\n"
    summary_text += f"  Specificity: {specificity[i]:.3f}\n"
    summary_text += f"  F1 Score: {f1[i]:.3f}\n"
    
    if i == 1 and specificity[i] > 0.9 and f1[i] < 0.1:
        summary_text += "  [PARTIAL COLLAPSE]\n"
        summary_text += "     Generator ignoring latent code\n"
    elif i == 2 and specificity[i] == 0 and accuracy[i] == 0.5:
        summary_text += "  [MODE COLLAPSE]\n"
        summary_text += "     Generator found cheat code\n"
    elif i == 0:
        summary_text += "  [WORKING]\n"
    
    summary_text += "\n"

summary_text += f"\nTraining: {epochs_trained[0]} epochs\n"
summary_text += f"Data source: {JSON_FILENAME}"

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace', linespacing=1.5,
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.suptitle('COLLAPSE ANALYSIS — IBM QPU Results', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ibm_collapse_analysis.png', dpi=150, bbox_inches='tight')
print(" Saved: figures/ibm_collapse_analysis.png")

# ==================== PRINT SUMMARY ====================
print("\n" + "="*70)
print(f"IBM QPU RESULTS SUMMARY — {JSON_FILENAME}")
print("="*70)
print(f"{'Features':>8} {'Qubits':>8} {'Epochs':>8} {'Accuracy':>10} {'Specificity':>12} {'F1':>8} {'StdMAE':>10} {'Status':>14}")
print("-"*80)

for i, r in enumerate(results):
    feat = r["n_features"]
    
    # Determine status
    if i == 0:
        status = "WORKING"
    elif i == 1 and specificity[i] > 0.9 and f1[i] < 0.1:
        status = "PARTIAL COLLAPSE"
    elif i == 2 and specificity[i] == 0 and accuracy[i] == 0.5:
        status = "MODE COLLAPSE"
    else:
        status = "LEARNING"
    
    print(f"{feat:>8} {qubits[i]:>8} {epochs_trained[i]:>8} "
          f"{accuracy[i]:>10.3f} {specificity[i]:>12.3f} {f1[i]:>8.3f} "
          f"{std_mae[i]:>10.4f} {status:>14}")

print("="*70)
print(f"\nAll figures saved to: figures/")
print(f"  • figures/ibm_results_complete.png")
print(f"  • figures/ibm_mae_comparison.png")
print(f"  • figures/ibm_training_curves.png")
print(f"  • figures/ibm_mae_convergence.png")
print(f"  • figures/ibm_collapse_analysis.png")