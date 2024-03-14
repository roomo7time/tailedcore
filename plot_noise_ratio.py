import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('./logs/tailedpatch_result_summary - noise_ratio.csv')

# Define column names as variables
tail_type_col = 'Tail type'
method_col = 'Method'
noise_ratio_col = 'Noise ratio'
image_level_mean_col = 'Image-level AUROC mean'
pixel_level_mean_col = 'Pixel-level AUROC mean'
image_level_std_col = 'Image-level AUROC std'
pixel_level_std_col = 'Pixel-level AUROC std'

# Set seaborn theme
sns.set_theme(style="whitegrid")

# Font size for labels and legend
font_size = 18
alpha = 0.1

# Get unique tail types
tail_types = data[tail_type_col].unique()

for tail_type in tail_types:
    # Filter data for the current tail type
    data_filtered = data[data[tail_type_col] == tail_type]
    methods = data_filtered[method_col].unique()
    # palette = sns.color_palette("viridis", n_colors=len(methods))
    # palette = sns.color_palette("hls", n_colors=len(methods))
    palette = sns.color_palette( n_colors=len(methods))


    # Plot for Image-level AUROC
    plt.figure()
    lineplot = sns.lineplot(data=data_filtered, x=noise_ratio_col, y=image_level_mean_col, hue=method_col, style=method_col, markers=True, dashes=False, errorbar=None, palette=palette, marker='o', linestyle='-', markersize=8)
    handles, labels = lineplot.get_legend_handles_labels()
    # Add shading for standard deviation
    for method, color in zip(methods, palette):
        subset = data_filtered[data_filtered[method_col] == method]
        plt.fill_between(subset[noise_ratio_col], subset[image_level_mean_col] - subset[image_level_std_col], 
                         subset[image_level_mean_col] + subset[image_level_std_col], color=color, alpha=alpha)
    plt.xlabel('Noise Ratio', fontsize=font_size)
    plt.ylabel('Image-level AUROC', fontsize=font_size)
    plt.title(tail_type, fontsize=font_size)
    plt.legend(handles=handles[0:], labels=labels[0:], fontsize=font_size, title_fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Image-level_AUROC_{tail_type}.pdf", bbox_inches='tight', dpi=300)
    plt.close()

    # Plot for Pixel-level AUROC
    plt.figure()
    lineplot = sns.lineplot(data=data_filtered, x=noise_ratio_col, y=pixel_level_mean_col, hue=method_col, style=method_col, markers=True, dashes=False, errorbar=None, palette=palette, marker='o', linestyle='-', markersize=8)
    handles, labels = lineplot.get_legend_handles_labels()
    # Add shading for standard deviation
    for method, color in zip(methods, palette):
        subset = data_filtered[data_filtered[method_col] == method]
        plt.fill_between(subset[noise_ratio_col], subset[pixel_level_mean_col] - subset[pixel_level_std_col], 
                         subset[pixel_level_mean_col] + subset[pixel_level_std_col], color=color, alpha=alpha)
    plt.xlabel('Noise Ratio', fontsize=font_size)
    plt.ylabel('Pixel-level AUROC', fontsize=font_size)
    plt.title(tail_type, fontsize=font_size)
    plt.legend(handles=handles[0:], labels=labels[0:], fontsize=font_size, title_fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Pixel-level_AUROC_{tail_type}.pdf", bbox_inches='tight', dpi=300)
    plt.close()