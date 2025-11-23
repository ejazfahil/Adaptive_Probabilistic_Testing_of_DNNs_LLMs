"""
Visualization Generator for DeepSample Framework Thesis
Creates all diagrams, charts, and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_framework_architecture():
    """Create DeepSample framework architecture diagram"""
    print("Creating framework architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'DeepSample Framework Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 7.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 8, 'Operational\nInput Pool', ha='center', va='center', fontweight='bold')
    
    # Auxiliary variables
    aux_box = FancyBboxPatch((4.5, 7.5), 3, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(aux_box)
    ax.text(6, 8, 'Auxiliary Variables\n(Conf, Entropy, SA)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Sampling strategies (center)
    strategies = [
        ('SRS', 1, 5.5),
        ('SUPS', 3, 5.5),
        ('RHC-S', 5, 5.5),
        ('SSRS', 7, 5.5),
        ('GBS', 9, 5.5),
        ('2-UPS', 11, 5.5),
        ('DeepEST', 13, 5.5)
    ]
    
    ax.text(7, 6.8, 'Sampling Strategies', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    for name, x, y in strategies:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='#E63946', facecolor='#F4A6A3', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows from input to strategies
    arrow1 = FancyArrowPatch((2, 7.5), (7, 6.2),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='#2E86AB')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((6, 7.5), (7, 6.2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#2E86AB')
    ax.add_patch(arrow2)
    
    # Sample selection
    sample_box = FancyBboxPatch((5, 3.8), 4, 0.8,
                               boxstyle="round,pad=0.1",
                               edgecolor='#F77F00', facecolor='#FCBF49', linewidth=2)
    ax.add_patch(sample_box)
    ax.text(7, 4.2, 'Selected Test Sample', ha='center', va='center', fontweight='bold')
    
    # Arrows from strategies to sample
    for name, x, y in strategies:
        arrow = FancyArrowPatch((x, y-0.3), (7, 4.6),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1, color='#E63946', alpha=0.6)
        ax.add_patch(arrow)
    
    # Estimation process
    est_box = FancyBboxPatch((2.5, 2.2), 3, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor='#06A77D', facecolor='#90E0C1', linewidth=2)
    ax.add_patch(est_box)
    ax.text(4, 2.7, 'Statistical\nEstimators', ha='center', va='center', fontweight='bold')
    
    # Failure detection
    fail_box = FancyBboxPatch((8.5, 2.2), 3, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='#06A77D', facecolor='#90E0C1', linewidth=2)
    ax.add_patch(fail_box)
    ax.text(10, 2.7, 'Misprediction\nDetection', ha='center', va='center', fontweight='bold')
    
    # Arrows to outputs
    arrow3 = FancyArrowPatch((6.5, 3.8), (4, 3.2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#F77F00')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((7.5, 3.8), (10, 3.2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#F77F00')
    ax.add_patch(arrow4)
    
    # Final outputs
    out1_box = FancyBboxPatch((2, 0.5), 3, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='#5A189A', facecolor='#C77DFF', linewidth=2)
    ax.add_patch(out1_box)
    ax.text(3.5, 0.9, 'Unbiased Accuracy\nEstimate', ha='center', va='center', fontweight='bold')
    
    out2_box = FancyBboxPatch((9, 0.5), 3, 0.8,
                             boxstyle="round,pad=0.1",
                             edgecolor='#5A189A', facecolor='#C77DFF', linewidth=2)
    ax.add_patch(out2_box)
    ax.text(10.5, 0.9, 'Exposed Model\nFailures', ha='center', va='center', fontweight='bold')
    
    # Final arrows
    arrow5 = FancyArrowPatch((4, 2.2), (3.5, 1.3),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#06A77D')
    ax.add_patch(arrow5)
    
    arrow6 = FancyArrowPatch((10, 2.2), (10.5, 1.3),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='#06A77D')
    ax.add_patch(arrow6)
    
    plt.tight_layout()
    plt.savefig('assets/deepsample_framework_architecture.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/deepsample_framework_architecture.png")
    plt.close()

def create_accuracy_comparison():
    """Create accuracy estimation comparison chart"""
    print("Creating accuracy estimation comparison...")
    
    # Load results
    df = pd.read_csv('datasets/sampling_results_comparison.csv')
    
    # Filter for sample size 200 and Confidence auxiliary
    df_plot = df[(df['sample_size'] == 200) & (df['auxiliary_variable'] == 'Confidence')]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['IMDb', 'SST-2', 'CIFAR-10']
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#5A189A', '#E9C46A', '#F4A261']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = df_plot[df_plot['dataset'] == dataset]
        
        methods = data['method'].values
        rmse = data['rmse'].values
        
        bars = ax.bar(range(len(methods)), rmse, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('RMSE', fontweight='bold')
        ax.set_title(f'{dataset} Dataset', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (method, val) in enumerate(zip(methods, rmse)):
            ax.text(i, val + 0.001, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Accuracy Estimation Performance (RMSE) - Sample Size: 200', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/accuracy_estimation_comparison.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/accuracy_estimation_comparison.png")
    plt.close()

def create_misprediction_detection():
    """Create misprediction detection rates chart"""
    print("Creating misprediction detection chart...")
    
    df = pd.read_csv('datasets/sampling_results_comparison.csv')
    df_plot = df[(df['sample_size'] == 200) & (df['auxiliary_variable'] == 'Confidence') & 
                 (df['dataset'] == 'IMDb')]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df_plot['method'].values
    failures = df_plot['failures_found'].values
    
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#5A189A', '#E9C46A', '#F4A261']
    bars = ax.barh(methods, failures, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Mispredictions Found', fontweight='bold', fontsize=12)
    ax.set_ylabel('Sampling Method', fontweight='bold', fontsize=12)
    ax.set_title('Misprediction Detection Performance\n(IMDb Dataset, Sample Size: 200)', 
                 fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (method, val) in enumerate(zip(methods, failures)):
        ax.text(val + 2, i, f'{int(val)}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('assets/misprediction_detection_rates.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/misprediction_detection_rates.png")
    plt.close()

def create_sample_size_sensitivity():
    """Create sample size sensitivity analysis"""
    print("Creating sample size sensitivity analysis...")
    
    df = pd.read_csv('datasets/sampling_results_comparison.csv')
    df_plot = df[(df['auxiliary_variable'] == 'Confidence') & (df['dataset'] == 'IMDb')]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df_plot['method'].unique()
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#5A189A', '#E9C46A', '#F4A261']
    
    for method, color in zip(methods, colors):
        data = df_plot[df_plot['method'] == method].sort_values('sample_size')
        ax.plot(data['sample_size'], data['rmse'], marker='o', linewidth=2.5, 
                label=method, color=color, markersize=8)
    
    ax.set_xlabel('Sample Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('RMSE', fontweight='bold', fontsize=12)
    ax.set_title('Sample Size Sensitivity Analysis\n(IMDb Dataset, Confidence Auxiliary)', 
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('assets/sample_size_sensitivity.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/sample_size_sensitivity.png")
    plt.close()

def create_tradeoffs_radar():
    """Create radar chart for method trade-offs"""
    print("Creating method trade-offs radar chart...")
    
    # Define metrics for each method (normalized 0-1)
    methods = ['SRS', 'SUPS', 'RHC-S', 'SSRS', 'GBS', '2-UPS', 'DeepEST']
    
    # Metrics: Accuracy Est., Failure Detection, Simplicity, Robustness, Efficiency
    metrics_data = {
        'SRS': [0.85, 0.25, 1.0, 0.9, 0.95],
        'SUPS': [0.75, 0.95, 0.7, 0.6, 0.8],
        'RHC-S': [0.80, 0.70, 0.6, 0.75, 0.75],
        'SSRS': [0.65, 0.90, 0.5, 0.65, 0.70],
        'GBS': [0.88, 0.60, 0.55, 0.85, 0.80],
        '2-UPS': [0.78, 0.65, 0.65, 0.70, 0.75],
        'DeepEST': [0.82, 0.85, 0.50, 0.75, 0.70]
    }
    
    categories = ['Accuracy\nEstimation', 'Failure\nDetection', 'Simplicity', 
                  'Robustness', 'Efficiency']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#5A189A', '#E9C46A', '#F4A261']
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[idx]
        values = metrics_data[method]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=method)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=7)
        ax.set_title(method, fontweight='bold', size=12, color=color, pad=20)
        ax.grid(True)
    
    # Remove the last empty subplot
    fig.delaxes(axes[-1])
    
    plt.suptitle('DeepSample Methods: Multi-Dimensional Performance Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('assets/method_tradeoffs_radar.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/method_tradeoffs_radar.png")
    plt.close()

def create_auxiliary_variables_scatter():
    """Create scatter plot of confidence vs entropy"""
    print("Creating confidence vs entropy scatter plot...")
    
    df = pd.read_csv('datasets/synthetic_sentiment_imdb.csv')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate correct and incorrect predictions
    correct = df[df['is_correct'] == 1]
    incorrect = df[df['is_correct'] == 0]
    
    ax.scatter(correct['confidence'], correct['entropy'], 
              alpha=0.5, s=20, c='#06A77D', label='Correct Predictions', edgecolors='none')
    ax.scatter(incorrect['confidence'], incorrect['entropy'], 
              alpha=0.7, s=30, c='#E63946', label='Mispredictions', 
              edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Confidence Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Prediction Entropy', fontweight='bold', fontsize=12)
    ax.set_title('Auxiliary Variables: Confidence vs Entropy\n(IMDb Dataset)', 
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/confidence_vs_entropy_scatter.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/confidence_vs_entropy_scatter.png")
    plt.close()

def create_estimation_error_boxplots():
    """Create boxplots of estimation errors"""
    print("Creating estimation error boxplots...")
    
    df = pd.read_csv('datasets/sampling_results_comparison.csv')
    df_plot = df[(df['auxiliary_variable'] == 'Confidence') & (df['dataset'] == 'IMDb')].copy()
    
    # Calculate estimation error
    df_plot.loc[:, 'estimation_error'] = df_plot['estimated_accuracy'] - df_plot['true_accuracy']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = ['SRS', 'SUPS', 'RHC-S', 'SSRS', 'GBS', '2-UPS', 'DeepEST']
    data_to_plot = [df_plot[df_plot['method'] == m]['estimation_error'].values for m in methods]
    
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#5A189A', '#E9C46A', '#F4A261']
    
    bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='black'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='True Accuracy')
    ax.set_ylabel('Estimation Error', fontweight='bold', fontsize=12)
    ax.set_xlabel('Sampling Method', fontweight='bold', fontsize=12)
    ax.set_title('Distribution of Accuracy Estimation Errors\n(IMDb Dataset, All Sample Sizes)', 
                 fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('assets/estimation_error_boxplots.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/estimation_error_boxplots.png")
    plt.close()

def create_sampling_strategies_flowchart():
    """Create decision flowchart for method selection"""
    print("Creating sampling strategies flowchart...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'DeepSample Method Selection Flowchart', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Start
    start = FancyBboxPatch((5.5, 8.2), 3, 0.6,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='#90E0C1', linewidth=2)
    ax.add_patch(start)
    ax.text(7, 8.5, 'Start: Testing Objective?', ha='center', va='center', fontweight='bold')
    
    # Decision 1
    dec1 = FancyBboxPatch((0.5, 6.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(dec1)
    ax.text(2, 7, 'Unbiased Accuracy\nEstimation Priority', ha='center', va='center', fontweight='bold')
    
    dec2 = FancyBboxPatch((5.5, 6.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(dec2)
    ax.text(7, 7, 'Failure Detection\nPriority', ha='center', va='center', fontweight='bold')
    
    dec3 = FancyBboxPatch((10.5, 6.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#2E86AB', facecolor='#A9D6E5', linewidth=2)
    ax.add_patch(dec3)
    ax.text(12, 7, 'Balanced\nApproach', ha='center', va='center', fontweight='bold')
    
    # Arrows from start
    arrow1 = FancyArrowPatch((6, 8.2), (2, 7.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((7, 8.2), (7, 7.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((8, 8.2), (12, 7.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Recommendations - Left branch
    rec1a = FancyBboxPatch((0.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#5A189A', facecolor='#C77DFF', linewidth=1.5)
    ax.add_patch(rec1a)
    ax.text(1.1, 5.2, 'SRS\n(Simple)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    rec1b = FancyBboxPatch((2.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#5A189A', facecolor='#C77DFF', linewidth=1.5)
    ax.add_patch(rec1b)
    ax.text(3.1, 5.2, 'GBS\n(Adaptive)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Middle branch
    rec2a = FancyBboxPatch((5.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#E63946', facecolor='#F4A6A3', linewidth=1.5)
    ax.add_patch(rec2a)
    ax.text(6.1, 5.2, 'SUPS\n(Aggressive)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    rec2b = FancyBboxPatch((7.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#E63946', facecolor='#F4A6A3', linewidth=1.5)
    ax.add_patch(rec2b)
    ax.text(8.1, 5.2, 'SSRS\n(Stratified)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Right branch
    rec3a = FancyBboxPatch((10.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#F77F00', facecolor='#FCBF49', linewidth=1.5)
    ax.add_patch(rec3a)
    ax.text(11.1, 5.2, 'RHC-S\n(Balanced)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    rec3b = FancyBboxPatch((12.2, 4.8), 1.8, 0.8,
                          boxstyle="round,pad=0.05",
                          edgecolor='#F77F00', facecolor='#FCBF49', linewidth=1.5)
    ax.add_patch(rec3b)
    ax.text(13.1, 5.2, 'DeepEST\n(Adaptive)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows to recommendations
    for x in [1.1, 3.1]:
        arrow = FancyArrowPatch((2, 6.5), (x, 5.6),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='#2E86AB')
        ax.add_patch(arrow)
    
    for x in [6.1, 8.1]:
        arrow = FancyArrowPatch((7, 6.5), (x, 5.6),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='#2E86AB')
        ax.add_patch(arrow)
    
    for x in [11.1, 13.1]:
        arrow = FancyArrowPatch((12, 6.5), (x, 5.6),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='#2E86AB')
        ax.add_patch(arrow)
    
    # Considerations box
    consid_box = FancyBboxPatch((3, 2.5), 8, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='#06A77D', facecolor='#E8F5E9', linewidth=2)
    ax.add_patch(consid_box)
    ax.text(7, 3.6, 'Additional Considerations:', ha='center', va='top', 
            fontweight='bold', fontsize=11)
    ax.text(7, 3.2, '• Sample Budget: Larger budgets → SRS/GBS; Smaller → SUPS/SSRS', 
            ha='center', va='top', fontsize=9)
    ax.text(7, 2.9, '• Auxiliary Variable: Confidence (convenient) vs Entropy/SA (robust)', 
            ha='center', va='top', fontsize=9)
    ax.text(7, 2.6, '• Domain: Vision (DSA/LSA) vs NLP (Confidence/Entropy)', 
            ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('assets/sampling_strategies_flowchart.png', bbox_inches='tight', dpi=300)
    print("✓ Saved: assets/sampling_strategies_flowchart.png")
    plt.close()

def main():
    """Generate all visualizations"""
    print("=" * 60)
    print("DeepSample Framework - Visualization Generation")
    print("=" * 60)
    
    create_framework_architecture()
    create_accuracy_comparison()
    create_misprediction_detection()
    create_sample_size_sensitivity()
    create_tradeoffs_radar()
    create_auxiliary_variables_scatter()
    create_estimation_error_boxplots()
    create_sampling_strategies_flowchart()
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
