import matplotlib.pyplot as plt
import numpy as np
from .metrics import calculate_entropy

def plot_entropy(df, feature, target_col='class'):
    """
    Plot entropy as rectangles where:
    - Height = entropy
    - Width = proportion of population
    
    Parameters:
    df: DataFrame with the data
    feature: column name to analyze
    target_col: name of the target/class column
    """
    # Get unique values and calculate entropy and proportions for each
    values = df[feature].unique()
    entropies = []
    proportions = []
    total = len(df)
    
    for value in values:
        subset = df[df[feature] == value][target_col]
        entropies.append(calculate_entropy(subset))
        proportions.append(len(subset) / total)
    
    # Sort by entropy (increasing order)
    sorted_indices = np.argsort(entropies)
    values_sorted = [values[i] for i in sorted_indices]
    entropies_sorted = [entropies[i] for i in sorted_indices]
    proportions_sorted = [proportions[i] for i in sorted_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate x positions (cumulative widths)
    x_start = 0
    
    for i, (value, entropy, proportion) in enumerate(zip(values_sorted, entropies_sorted, proportions_sorted)):
        # Draw rectangle
        rect = plt.Rectangle((x_start, 0), proportion, entropy, 
                            facecolor='steelblue', 
                            edgecolor='black', 
                            linewidth=2,
                            alpha=0.7)
        ax.add_patch(rect)
        
        # Add label in center of rectangle
        center_x = x_start + proportion / 2
        center_y = entropy / 2

        
        # # Add proportion label at bottom
        # ax.text(center_x, -0.05, f'{proportion:.1%}', 
        #        ha='center', va='top', fontsize=9, color='gray')
        circle = plt.Circle((center_x, center_y), 0.025, 
                    facecolor='white', 
                    edgecolor='black', 
                    linewidth=1.5,
                    zorder=10)
        ax.add_patch(circle)
        ax.text(center_x, center_y, f'{value}', 
               ha='center', va='center', fontsize=9, fontweight='bold',
               zorder=11)

        x_start += proportion
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Proportion of Population', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title(f'Entropy and prevalence values for {feature.upper()}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_aspect('equal', adjustable='box')  # This makes circles truly circular!

    plt.tight_layout()
    plt.show()