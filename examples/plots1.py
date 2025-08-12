#!/usr/bin/env python3
"""
Multi-Lambda Contextual Fraction vs Sigma Analysis

This script analyzes how the scaled contextual fraction (CF/λ²) varies with
sigma for different lambda values. It creates comparative visualizations showing
how the contextual fraction scaling behavior changes across different coupling strengths.

Key features:
- Multiple lambda values for comparison
- Different coupling strength regimes (weak to strong)
- Systematic sigma variation
- Comparative contextual fraction vs sigma visualizations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qft.detectors import rho, QregDelta, QregHeavsde, Qmagic
    from optimization.lin_prog import contextual_fraction
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure src/ directory contains the required modules.")
    sys.exit(1)


def analyze_single_state(sigma, d, a, lam, qfunc):
    """
    Analyze a single quantum state for contextual fraction.
    
    Args:
        sigma, d, a, lam: Physical parameters
        qfunc: Q-function (QregDelta, QregHeavsde, or Qmagic)
    
    Returns:
        dict: Analysis results including CF, CF/λ², and validity checks
    """
    try:
        # Generate the density matrix
        rho_matrix = rho(sigma, d, a, qfunc, lam)
        
        # Validate density matrix properties
        trace = np.trace(rho_matrix)
        eigenvals = np.linalg.eigvals(rho_matrix)
        min_eigenval = np.min(np.real(eigenvals))
        is_hermitian = np.allclose(rho_matrix, rho_matrix.T.conj())
        
        # Check if it's a valid density matrix
        is_valid = (abs(trace - 1.0) < 1e-10 and
                    is_hermitian and
                    min_eigenval >= -1e-10)
        
        if not is_valid:
            return {
                'success': False,
                'error': 'Invalid density matrix',
                'trace': trace,
                'min_eigenval': min_eigenval
            }
        
        # Calculate contextual fraction
        cf_result = contextual_fraction(rho_matrix)
        
        if not cf_result['success']:
            return {
                'success': False,
                'error': 'CF calculation failed',
                'trace': trace,
                'min_eigenval': min_eigenval
            }
        
        cf_value = cf_result['b']
        cf_scaled = cf_value / (lam**2) if lam > 0 else 0.0
        
        return {
            'success': True,
            'cf': cf_value,
            'cf_scaled': cf_scaled,
            'trace': trace,
            'min_eigenval': min_eigenval,
            'is_valid': is_valid
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Exception: {str(e)}'}


def scan_sigma_for_ad_ratio(a_d_ratio, sigma_values, lam, qfunc_name):
    """
    Scan sigma values for a fixed a/d ratio.
    
    Args:
        a_d_ratio: Fixed ratio a/d
        sigma_values: Array of sigma values to scan
        lam: Coupling strength
        qfunc_name: Name of Q-function ('QregDelta', 'QregHeavsde', 'Qmagic')
    
    Returns:
        dict: Results containing arrays of sigma, CF, CF/λ², and success flags
    """
    # Map function names to functions
    qfunc_map = {
        'QregDelta': QregDelta,
        'QregHeavsde': QregHeavsde,
        'Qmagic': Qmagic
    }
    qfunc = qfunc_map[qfunc_name]
    
    results = {
        'sigma_values': [],
        'cf_values': [],
        'cf_scaled_values': [],
        'd_values': [],
        'a_values': [],
        'success_flags': []
    }
    
    for sigma in sigma_values:
        # Choose d to maintain d >> sigma while keeping numerical stability
        # Use a scaling that ensures d/sigma ~ 5-10 for good physics
        d = max(10.0, 8.0 * sigma)  # Ensures d >> sigma
        a = a_d_ratio * d
        
        result = analyze_single_state(sigma, d, a, lam, qfunc)
        
        # Store parameters
        results['sigma_values'].append(sigma)
        results['d_values'].append(d)
        results['a_values'].append(a)
        
        # Store results
        if result['success']:
            results['cf_values'].append(result['cf'])
            results['cf_scaled_values'].append(result['cf_scaled'])
            results['success_flags'].append(True)
        else:
            results['cf_values'].append(np.nan)
            results['cf_scaled_values'].append(np.nan)
            results['success_flags'].append(False)
    
    return results


def create_multi_lambda_plots(all_lambda_results, sigma_values):
    """
    Create comparative plots for different lambda values.
    
    Args:
        all_lambda_results: Dictionary with structure 
                           {lambda_val: {ratio_name: {qfunc_name: scan_results}}}
        sigma_values: Array of sigma values used
    """
    # Set up subplots for different comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for different lambda values
    lambda_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    qfunc_markers = {'QregDelta': 'o', 'QregHeavsde': 's', 'Qmagic': '^'}
    
    # Plot 1: CF/λ² vs σ for different λ (QregHeavsde, a/d=1.0)
    ax1 = axes[0, 0]
    for i, (lam_val, lam_results) in enumerate(all_lambda_results.items()):
        if 'a/d = 1.0' in lam_results and 'QregHeavsde' in lam_results['a/d = 1.0']:
            scan_data = lam_results['a/d = 1.0']['QregHeavsde']
            sigma_vals = np.array(scan_data['sigma_values'])
            cf_scaled_vals = np.array(scan_data['cf_scaled_values'])
            success_flags = np.array(scan_data['success_flags'])
            
            valid_mask = (success_flags & 
                         ~np.isnan(cf_scaled_vals) & 
                         (cf_scaled_vals > 0))
            
            if np.any(valid_mask):
                color = lambda_colors[i % len(lambda_colors)]
                ax1.plot(sigma_vals[valid_mask], cf_scaled_vals[valid_mask],
                        'o-', color=color, alpha=0.7, 
                        label=f'λ = {lam_val}', markersize=6, linewidth=2)
    
    ax1.set_xlabel('σ', fontsize=12)
    ax1.set_ylabel('CF/λ²', fontsize=12)
    ax1.set_title('CF/λ² vs σ for Different λ (QregHeavsde, a/d=1.0)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Plot 2: Raw CF vs σ for different λ (QregHeavsde, a/d=1.0)
    ax2 = axes[0, 1]
    for i, (lam_val, lam_results) in enumerate(all_lambda_results.items()):
        if 'a/d = 1.0' in lam_results and 'QregHeavsde' in lam_results['a/d = 1.0']:
            scan_data = lam_results['a/d = 1.0']['QregHeavsde']
            sigma_vals = np.array(scan_data['sigma_values'])
            cf_vals = np.array(scan_data['cf_values'])
            success_flags = np.array(scan_data['success_flags'])
            
            valid_mask = (success_flags & 
                         ~np.isnan(cf_vals) & 
                         (cf_vals > 0))
            
            if np.any(valid_mask):
                color = lambda_colors[i % len(lambda_colors)]
                ax2.plot(sigma_vals[valid_mask], cf_vals[valid_mask],
                        's-', color=color, alpha=0.7, 
                        label=f'λ = {lam_val}', markersize=6, linewidth=2)
    
    ax2.set_xlabel('σ', fontsize=12)
    ax2.set_ylabel('Raw Contextual Fraction', fontsize=12)
    ax2.set_title('Raw CF vs σ for Different λ (QregHeavsde, a/d=1.0)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # Plot 3: CF/λ² vs σ for all Q-functions (λ = middle value, a/d=1.0)
    ax3 = axes[1, 0]
    lambda_values = list(all_lambda_results.keys())
    middle_lambda = lambda_values[len(lambda_values)//2]
    
    if middle_lambda in all_lambda_results and 'a/d = 1.0' in all_lambda_results[middle_lambda]:
        for qfunc_name in ['QregDelta', 'QregHeavsde', 'Qmagic']:
            if qfunc_name in all_lambda_results[middle_lambda]['a/d = 1.0']:
                scan_data = all_lambda_results[middle_lambda]['a/d = 1.0'][qfunc_name]
                sigma_vals = np.array(scan_data['sigma_values'])
                cf_scaled_vals = np.array(scan_data['cf_scaled_values'])
                success_flags = np.array(scan_data['success_flags'])
                
                valid_mask = (success_flags & 
                             ~np.isnan(cf_scaled_vals) & 
                             (cf_scaled_vals > 0))
                
                if np.any(valid_mask):
                    marker = qfunc_markers[qfunc_name]
                    ax3.plot(sigma_vals[valid_mask], cf_scaled_vals[valid_mask],
                            marker + '-', alpha=0.7, 
                            label=f'{qfunc_name}', markersize=6, linewidth=2)
    
    ax3.set_xlabel('σ', fontsize=12)
    ax3.set_ylabel('CF/λ²', fontsize=12)
    ax3.set_title(f'CF/λ² vs σ for Different Q-functions (λ={middle_lambda}, a/d=1.0)', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    
    # Plot 4: CF/λ² scaling with λ at fixed σ
    ax4 = axes[1, 1]
    lambda_vals = list(all_lambda_results.keys())
    # Pick a few representative sigma values
    representative_sigmas = [sigma_values[3], sigma_values[6], sigma_values[9]]
    
    for sigma_idx, target_sigma in enumerate(representative_sigmas):
        cf_scaled_at_sigma = []
        lambda_plot_vals = []
        
        for lam_val in lambda_vals:
            if ('a/d = 1.0' in all_lambda_results[lam_val] and 
                'QregHeavsde' in all_lambda_results[lam_val]['a/d = 1.0']):
                
                scan_data = all_lambda_results[lam_val]['a/d = 1.0']['QregHeavsde']
                sigma_vals = np.array(scan_data['sigma_values'])
                cf_scaled_vals = np.array(scan_data['cf_scaled_values'])
                success_flags = np.array(scan_data['success_flags'])
                
                # Find closest sigma value
                sigma_diff = np.abs(sigma_vals - target_sigma)
                closest_idx = np.argmin(sigma_diff)
                
                if (success_flags[closest_idx] and 
                    not np.isnan(cf_scaled_vals[closest_idx]) and
                    cf_scaled_vals[closest_idx] > 0):
                    cf_scaled_at_sigma.append(cf_scaled_vals[closest_idx])
                    lambda_plot_vals.append(lam_val)
        
        if len(cf_scaled_at_sigma) > 1:
            ax4.plot(lambda_plot_vals, cf_scaled_at_sigma, 
                    'o-', alpha=0.7, markersize=6, linewidth=2,
                    label=f'σ ≈ {target_sigma:.2f}')
    
    ax4.set_xlabel('λ', fontsize=12)
    ax4.set_ylabel('CF/λ²', fontsize=12)
    ax4.set_title('CF/λ² vs λ at Fixed σ (QregHeavsde, a/d=1.0)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('multi_lambda_contextual_fraction_analysis.png', dpi=300, 
                bbox_inches='tight')
    plt.show()


def main():
    """
    Main analysis function for multi-lambda contextual fraction analysis.
    """
    print("Multi-Lambda Analysis: CF/λ² vs σ for Different λ Values")
    print("=" * 60)
    print("Exploring contextual fraction scaling across coupling regimes")
    print()
    
    # Multiple lambda values to compare different coupling strengths
    lambda_values = [0.005, 0.01, 0.02, 0.04, 0.08]  # Weak to strong coupling
    
    # Sigma range - focus on regime where calculations are stable
    sigma_values = np.logspace(-0.5, 0.7, 12)  # ~0.32 to ~5.0
    
    # Simplified a/d ratios (focus on key cases)
    a_d_ratios = {
        'a/d = 0.8': 0.8,
        'a/d = 1.0': 1.0,
        'a/d = 1.2': 1.2
    }
    
    # Q-functions to analyze
    qfunctions = ['QregDelta', 'QregHeavsde', 'Qmagic']
    
    print("Analysis parameters:")
    print(f"  λ values: {lambda_values}")
    print(f"  σ range: [{sigma_values[0]:.2f}, {sigma_values[-1]:.2f}]")
    print(f"  a/d ratios: {list(a_d_ratios.keys())}")
    print(f"  Q-functions: {qfunctions}")
    print(f"  d scaling: d = max(10, 8σ) to ensure d >> σ")
    print()
    
    # Store results for all lambda values
    all_lambda_results = {}
    
    total_calculations = (len(lambda_values) * len(a_d_ratios) * 
                         len(qfunctions) * len(sigma_values))
    current_calc = 0
    
    for lam in lambda_values:
        print(f"Analyzing λ = {lam}...")
        all_lambda_results[lam] = {}
        
        for ratio_name, a_d_ratio in a_d_ratios.items():
            print(f"  {ratio_name}...")
            all_lambda_results[lam][ratio_name] = {}
            
            for qfunc_name in qfunctions:
                print(f"    Q-function: {qfunc_name}")
                
                # Perform sigma scan
                scan_results = scan_sigma_for_ad_ratio(a_d_ratio, sigma_values, 
                                                      lam, qfunc_name)
                all_lambda_results[lam][ratio_name][qfunc_name] = scan_results
                
                # Report progress and success rate
                success_count = sum(scan_results['success_flags'])
                total_points = len(scan_results['success_flags'])
                success_rate = 100 * success_count / total_points
                
                print(f"      Success: {success_count}/{total_points} " +
                      f"({success_rate:.1f}%)")
                
                current_calc += total_points
                overall_progress = 100 * current_calc / total_calculations
                print(f"      Overall progress: {overall_progress:.1f}%")
            
            print()
        print()
    
    # Create multi-lambda comparative visualization
    print("Creating multi-lambda comparative plots...")
    create_multi_lambda_plots(all_lambda_results, sigma_values)
    
    # Summary analysis across all lambda values
    print("\nSummary of CF/λ² Results Across All λ:")
    print("-" * 55)
    
    for lam in lambda_values:
        print(f"\nλ = {lam}:")
        lam_results = all_lambda_results[lam]
        
        for ratio_name, ratio_data in lam_results.items():
            print(f"  {ratio_name}:")
            
            for qfunc_name, scan_data in ratio_data.items():
                cf_scaled_vals = np.array(scan_data['cf_scaled_values'])
                valid_vals = cf_scaled_vals[~np.isnan(cf_scaled_vals) & 
                                          (cf_scaled_vals > 0)]
                
                if len(valid_vals) > 0:
                    min_val = np.min(valid_vals)
                    max_val = np.max(valid_vals)
                    mean_val = np.mean(valid_vals)
                    print(f"    {qfunc_name:12}: CF/λ² ∈ [{min_val:.4f}, " +
                          f"{max_val:.4f}], mean = {mean_val:.4f}")
                else:
                    print(f"    {qfunc_name:12}: No valid positive results")
    
    # Find optimal lambda for each Q-function
    print(f"\nOptimal λ for Maximum CF/λ² (a/d=1.0):")
    print("-" * 45)
    
    for qfunc_name in qfunctions:
        best_cf_scaled = 0
        best_lambda = None
        best_sigma = None
        
        for lam in lambda_values:
            if ('a/d = 1.0' in all_lambda_results[lam] and 
                qfunc_name in all_lambda_results[lam]['a/d = 1.0']):
                
                scan_data = all_lambda_results[lam]['a/d = 1.0'][qfunc_name]
                cf_scaled_vals = np.array(scan_data['cf_scaled_values'])
                sigma_vals = np.array(scan_data['sigma_values'])
                success_flags = np.array(scan_data['success_flags'])
                
                valid_mask = (success_flags & 
                             ~np.isnan(cf_scaled_vals) & 
                             (cf_scaled_vals > 0))
                
                if np.any(valid_mask):
                    max_cf_scaled = np.max(cf_scaled_vals[valid_mask])
                    if max_cf_scaled > best_cf_scaled:
                        best_cf_scaled = max_cf_scaled
                        best_lambda = lam
                        max_idx = np.argmax(cf_scaled_vals[valid_mask])
                        valid_indices = np.where(valid_mask)[0]
                        actual_idx = valid_indices[max_idx]
                        best_sigma = sigma_vals[actual_idx]
        
        if best_lambda is not None:
            print(f"{qfunc_name:12}: λ = {best_lambda}, " +
                  f"CF/λ² = {best_cf_scaled:.4f} at σ = {best_sigma:.3f}")
        else:
            print(f"{qfunc_name:12}: No valid results found")
    
    print(f"\nVisualization saved as 'multi_lambda_contextual_fraction_analysis.png'")
    print("Multi-lambda analysis complete!")


if __name__ == "__main__":
    main()
