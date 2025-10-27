import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cramers_v(confusion_matrix):
    """Calculate Cramer's V from a confusion matrix"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def bootstrap_chi_square_analysis(data, target_var, n_bootstrap=1000):
    """
    Perform bootstrap analysis with chi-square tests and Cramer's V
    """
    # Convert numeric variables with <=10 unique values to categorical
    data_processed = data.copy()
    for col in data_processed.select_dtypes(include=[np.number]).columns:
        if data_processed[col].nunique() <= 10:
            data_processed[col] = data_processed[col].astype('category')
    
    # Identify categorical variables (excluding target)
    categorical_vars = [col for col in data_processed.columns 
                      if data_processed[col].dtype.name == 'category' and col != target_var]
    
    print(f"Found {len(categorical_vars)} categorical variables: {categorical_vars}")
    
    # Storage for results
    results = {}
    
    for var in categorical_vars:
        print(f"Processing variable: {var}")
        
        var_results = {
            'chi_square_values': [],
            'p_values': [],
            'cramers_v_values': [],
            'degrees_freedom': []
        }
        
        # Bootstrap sampling
        for i in range(n_bootstrap):
            # Create bootstrap sample
            bootstrap_sample = resample(data_processed, random_state=i)
            
            # Create contingency table
            contingency_table = pd.crosstab(bootstrap_sample[var], bootstrap_sample[target_var]) # type: ignore
            
            # Skip if table is degenerate
            if min(contingency_table.shape) < 2 or contingency_table.sum().sum() == 0:
                continue
                
            # Perform chi-square test
            try:
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                cramers_v_val = cramers_v(contingency_table.values)
                
                var_results['chi_square_values'].append(chi2)
                var_results['p_values'].append(p_val)
                var_results['cramers_v_values'].append(cramers_v_val)
                var_results['degrees_freedom'].append(dof)
                
            except Exception as e:
                print(f"Error in bootstrap {i} for variable {var}: {e}")
                continue
        
        results[var] = var_results
    
    return results

def create_summary_table(results):
    """Create summary table with counts and average values"""
    summary_data = []
    
    for var, var_results in results.items():
        if len(var_results['chi_square_values']) > 0:
            p_values = np.array(var_results['p_values'])
            significant_count = np.sum(p_values < 0.05)
            total_samples = len(p_values)
            significant_percentage = (significant_count / total_samples) * 100 if total_samples > 0 else 0
            
            summary_data.append({
                'Variable': var,
                'Avg_Chi_Square': np.mean(var_results['chi_square_values']),
                'Avg_Cramers_V': np.mean(var_results['cramers_v_values']),
                'P_Values_Under_005_Count': significant_count,
                'P_Values_005_and_Above_Count': np.sum(p_values >= 0.05),
                'Significant_Percentage': significant_percentage,
                'Total_Samples': total_samples
            })
    
    return pd.DataFrame(summary_data)

def plot_p_value_distributions(results, save_plots=True):
    """Create plots for p-values >= 0.05 and < 0.05"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('P-Value Distributions from Bootstrap Analysis', fontsize=16, fontweight='bold')
    
    # Collect all p-values
    all_p_values = []
    var_labels = []
    
    for var, var_results in results.items():
        p_vals = var_results['p_values']
        all_p_values.extend(p_vals)
        var_labels.extend([var] * len(p_vals))
    
    all_p_values = np.array(all_p_values)
    
    # Plot 1: Histogram of all p-values
    axes[0,0].hist(all_p_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    axes[0,0].set_xlabel('P-Values')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of All P-Values')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: P-values < 0.05
    significant_p = all_p_values[all_p_values < 0.05]
    if len(significant_p) > 0:
        axes[0,1].hist(significant_p, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_xlabel('P-Values')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title(f'P-Values < 0.05 (n={len(significant_p)})')
        axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5, 0.5, 'No significant p-values', ha='center', va='center', transform=axes[0,1].transAxes)
    
    # Plot 3: P-values >= 0.05
    non_significant_p = all_p_values[all_p_values >= 0.05]
    if len(non_significant_p) > 0:
        axes[1,0].hist(non_significant_p, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_xlabel('P-Values')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'P-Values ≥ 0.05 (n={len(non_significant_p)})')
        axes[1,0].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No non-significant p-values', ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Plot 4: Summary statistics or simplified visualization
    num_vars = len(results)
    if num_vars <= 15:  # Show box plot only if manageable number of variables
        p_val_df = pd.DataFrame({'Variable': var_labels, 'P_Value': all_p_values})
        sns.boxplot(data=p_val_df, x='Variable', y='P_Value', ax=axes[1,1])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha='right')
        axes[1,1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
        axes[1,1].set_title('P-Value Distribution by Variable')
    else:
        # Show summary statistics instead of box plot
        var_sig_counts = []
        var_names_short = []
        
        for var, var_results in results.items():
            p_vals = np.array(var_results['p_values'])
            sig_count = np.sum(p_vals < 0.05)
            total_count = len(p_vals)
            sig_percentage = (sig_count / total_count) * 100 if total_count > 0 else 0
            var_sig_counts.append(sig_percentage)
            # Truncate long variable names
            var_name = var[:15] + '...' if len(var) > 18 else var
            var_names_short.append(var_name)
        
        # Create histogram of significance percentages
        axes[1,1].hist(var_sig_counts, bins=20, alpha=0.7, color='gold', edgecolor='orange')
        axes[1,1].axvline(x=5, color='red', linestyle='--', alpha=0.7, label='5% significance')
        axes[1,1].set_xlabel('Percentage of Significant Tests (%)')
        axes[1,1].set_ylabel('Number of Variables')
        axes[1,1].set_title(f'Distribution of Significance Rates\n({num_vars} variables)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add summary text
        median_sig = np.median(var_sig_counts)
        mean_sig = np.mean(var_sig_counts)
        axes[1,1].text(0.7, 0.8, f'Median: {median_sig:.1f}%\nMean: {mean_sig:.1f}%', 
                      transform=axes[1,1].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('p_value_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # If there are many variables, create an additional detailed significance plot
    if num_vars > 20:
        # Create a separate plot showing significance rates for each variable
        var_sig_data = []
        for var, var_results in results.items():
            p_vals = np.array(var_results['p_values'])
            sig_count = np.sum(p_vals < 0.05)
            total_count = len(p_vals)
            sig_percentage = (sig_count / total_count) * 100 if total_count > 0 else 0
            var_sig_data.append((var, sig_percentage, sig_count, total_count))
        
        # Sort by significance percentage
        var_sig_data.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 30 most significant variables
        top_n = min(30, len(var_sig_data))
        top_vars = [item[0] for item in var_sig_data[:top_n]]
        top_percentages = [item[1] for item in var_sig_data[:top_n]]
        
        plt.figure(figsize=(12, max(8, top_n * 0.3)))
        y_pos = np.arange(len(top_vars))
        bars = plt.barh(y_pos, top_percentages, color='lightcoral', alpha=0.7, edgecolor='darkred')
        plt.yticks(y_pos, [var[:25] + '...' if len(var) > 28 else var for var in top_vars], fontsize=8)
        plt.xlabel('Percentage of Significant Tests (%)')
        plt.title(f'Top {top_n} Variables by Significance Rate', fontsize=14, fontweight='bold')
        plt.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_percentages)):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('top_significant_variables.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Created additional plot showing top {top_n} most significant variables.")

def plot_average_values(results, save_plots=True):
    """Create plots for average Chi-square and Cramer's V values"""
    # Prepare data
    variables = []
    avg_chi_square = []
    avg_cramers_v = []
    
    for var, var_results in results.items():
        if len(var_results['chi_square_values']) > 0:
            variables.append(var)
            avg_chi_square.append(np.mean(var_results['chi_square_values']))
            avg_cramers_v.append(np.mean(var_results['cramers_v_values']))
    
    num_vars = len(variables)
    print(f"Creating plots for {num_vars} variables...")
    
    # If there are many variables, create separate large plots
    if num_vars > 20:
        # Create individual large plots for better readability
        
        # Plot 1: Chi-square values (horizontal bar chart for better label readability)
        fig1, ax1 = plt.subplots(figsize=(12, max(8, num_vars * 0.3)))
        y_pos = np.arange(len(variables))
        bars1 = ax1.barh(y_pos, avg_chi_square, color='lightblue', edgecolor='navy', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(variables, fontsize=8)
        ax1.set_xlabel('Average Chi-Square Value')
        ax1.set_title('Average Chi-Square Values by Variable', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, avg_chi_square)):
            ax1.text(bar.get_width() + max(avg_chi_square)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('chi_square_values.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Cramer's V values (horizontal bar chart)
        fig2, ax2 = plt.subplots(figsize=(12, max(8, num_vars * 0.3)))
        bars2 = ax2.barh(y_pos, avg_cramers_v, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(variables, fontsize=8)
        ax2.set_xlabel('Average Cramer\'s V Value')
        ax2.set_title('Average Cramer\'s V Values by Variable', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)  # Cramer's V ranges from 0 to 1
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars2, avg_cramers_v)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('cramers_v_values.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Scatter plot (no labels to avoid clutter)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        scatter = ax3.scatter(avg_chi_square, avg_cramers_v, s=60, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Average Chi-Square Value')
        ax3.set_ylabel('Average Cramer\'s V Value')
        ax3.set_title('Chi-Square vs Cramer\'s V Relationship', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add text box with correlation info
        correlation = np.corrcoef(avg_chi_square, avg_cramers_v)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('chi_square_vs_cramers_v.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot 4: Vertical scatter plot of Chi-square vs Cramer's V with enhanced features
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot with color coding based on Cramer's V values
        scatter = ax4.scatter(avg_chi_square, avg_cramers_v, s=80, alpha=0.7, 
                             c=avg_cramers_v, cmap='viridis', edgecolor='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Cramer\'s V Value', rotation=270, labelpad=20)
        
        ax4.set_xlabel('Average Chi-Square Value', fontsize=12)
        ax4.set_ylabel('Average Cramer\'s V Value', fontsize=12)
        ax4.set_title('Chi-Square vs Cramer\'s V Relationship\n(All Variables)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation info
        correlation = np.corrcoef(avg_chi_square, avg_cramers_v)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nTotal Variables: {num_vars}', 
                transform=ax4.transAxes, fontsize=11,
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})
        
        # Highlight top variables with annotations (top 10 by Cramer's V)
        sorted_indices = np.argsort(avg_cramers_v)[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            var_name = variables[idx][:20] + '...' if len(variables[idx]) > 23 else variables[idx]
            ax4.annotate(var_name, 
                        (avg_chi_square[idx], avg_cramers_v[idx]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8,
                        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'lightblue', 'alpha': 0.7})
        
        # Add reference lines
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Weak association (0.1)')
        ax4.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate association (0.3)')
        ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong association (0.5)')
        ax4.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('chi_square_vs_cramers_v_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Original 2x2 subplot layout for smaller number of variables
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Average Statistical Values from Bootstrap Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Average Chi-square values
        bars1 = axes[0,0].bar(variables, avg_chi_square, color='lightblue', edgecolor='navy', alpha=0.7)
        axes[0,0].set_xlabel('Variables')
        axes[0,0].set_ylabel('Average Chi-Square Value')
        axes[0,0].set_title('Average Chi-Square Values by Variable')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Average Cramer's V values
        bars2 = axes[0,1].bar(variables, avg_cramers_v, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[0,1].set_xlabel('Variables')
        axes[0,1].set_ylabel('Average Cramer\'s V Value')
        axes[0,1].set_title('Average Cramer\'s V Values by Variable')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot
        axes[1,0].scatter(avg_chi_square, avg_cramers_v, s=100, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].set_xlabel('Average Chi-Square Value')
        axes[1,0].set_ylabel('Average Cramer\'s V Value')
        axes[1,0].set_title('Chi-Square vs Cramer\'s V Relationship')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Combined bar chart
        x = np.arange(len(variables))
        width = 0.35
        normalized_chi = np.array(avg_chi_square) / max(avg_chi_square) if max(avg_chi_square) > 0 else avg_chi_square
        
        axes[1,1].bar(x - width/2, normalized_chi, width, label='Normalized Chi-Square', 
                     color='lightblue', alpha=0.7, edgecolor='navy')
        axes[1,1].bar(x + width/2, avg_cramers_v, width, label='Cramer\'s V', 
                     color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        axes[1,1].set_xlabel('Variables')
        axes[1,1].set_ylabel('Value')
        axes[1,1].set_title('Normalized Chi-Square vs Cramer\'s V Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(variables, rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('average_values_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the complete analysis"""
    # Load data
    print("Loading data...")
    try:
        # Replace 'training_data.csv' with your actual file path
        data = pd.read_csv('training_data.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found. Please ensure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Set target variable
    target_var = 'risk_level'
    
    if target_var not in data.columns:
        print(f"Error: Target variable '{target_var}' not found in the dataset.")
        print(f"Available columns: {list(data.columns)}")
        return
    
    print(f"Target variable: {target_var}")
    print(f"Target variable unique values: {data[target_var].unique()}")
    
    # Perform bootstrap analysis
    print("\nPerforming bootstrap analysis...")
    results = bootstrap_chi_square_analysis(data, target_var, n_bootstrap=1000)
    
    if not results:
        print("No results generated. Please check your data and target variable.")
        return
    
    # Create and display summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(results)
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Save summary table
    summary_df.to_csv('bootstrap_summary_table.csv', index=False)
    print("\nSummary table saved as 'bootstrap_summary_table.csv'")
    
    # Create plots
    print("\nCreating plots...")
    plot_p_value_distributions(results)
    plot_average_values(results)
    
    # Print overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    all_chi_squares = [val for var_results in results.values() for val in var_results['chi_square_values']]
    all_cramers_v = [val for var_results in results.values() for val in var_results['cramers_v_values']]
    all_p_values = [val for var_results in results.values() for val in var_results['p_values']]
    
    print(f"Overall Average Chi-Square: {np.mean(all_chi_squares):.4f}")
    print(f"Overall Average Cramer's V: {np.mean(all_cramers_v):.4f}")
    print(f"Overall Average P-Value: {np.mean(all_p_values):.4f}")
    print(f"Percentage of Significant Tests (p < 0.05): {100 * np.sum(np.array(all_p_values) < 0.05) / len(all_p_values):.2f}%")
    print(f"Total Bootstrap Samples: {len(all_p_values)}")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()