#!/usr/bin/env python3
"""
Scaled Contextual Fraction Contour Analysis: CF/λ² vs a and d

This script creates a single contour plot of scaled contextual fraction (CF/λ²) 
as a function of detector parameters a and d for fixed lambda and sigma values.

Key features:
- Fixed σ = 1.4 and λ = 0.02
- Systematic scanning of a and d parameter space
- Single contour visualization with heat map of CF/λ²
- QregHeavsde Q-function focus
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from qft.detectors import rho, QregHeavsde
    from optimization.lin_prog import contextual_fraction
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure src/ directory contains the required modules.")
    sys.exit(1)


def analyze_single_point(sigma, d, a, lam, qfunc):
    """
    Analyze contextual fraction at a single point in (a,d) space.
    
    Args:
        sigma: Field correlation parameter
        d: Detector separation
        a: Detector size parameter
        lam: Coupling strength
        qfunc: Q-function (typically QregHeavsde)
    
    Returns:
        float: Scaled contextual fraction value (CF/λ²) (NaN if calculation fails)
    """
    try:
        # Generate the density matrix
        rho_matrix = rho(sigma, d, a, qfunc, lam)
        
        # Basic validation
        trace = np.trace(rho_matrix)
        eigenvals = np.linalg.eigvals(rho_matrix)
        min_eigenval = np.min(np.real(eigenvals))
        is_hermitian = np.allclose(rho_matrix, rho_matrix.T.conj())
        
        # Check if it's a valid density matrix
        is_valid = (abs(trace - 1.0) < 1e-10 and
                    is_hermitian and
                    min_eigenval >= -1e-10)
        
        if not is_valid:
            return np.nan
        
        # Calculate contextual fraction
        cf_result = contextual_fraction(rho_matrix)
        
        if not cf_result['success']:
            return np.nan
        
        # Return scaled contextual fraction (CF/λ²)
        cf_scaled = cf_result['b'] / (lam**2) if lam > 0 else np.nan
        return cf_scaled
        
    except Exception:
        return np.nan


def scan_a_d_space(sigma, lam, a_range, d_range, qfunc=QregHeavsde):
    """
    Scan the (a,d) parameter space for scaled contextual fraction.
    
    Args:
        sigma: Field correlation parameter
        lam: Coupling strength
        a_range: Array of a values to scan
        d_range: Array of d values to scan
        qfunc: Q-function to use
    
    Returns:
        tuple: (A_grid, D_grid, CF_scaled_grid) where grids are 2D arrays
    """
    print(f"  Scanning (a,d) space for σ = {sigma:.3f}, λ = {lam:.3f}...")
    
    # Create coordinate grids
    A_grid, D_grid = np.meshgrid(a_range, d_range)
    CF_scaled_grid = np.zeros_like(A_grid)
    
    total_points = len(a_range) * len(d_range)
    current_point = 0
    
    # Scan over all (a,d) combinations
    for i, d in enumerate(d_range):
        for j, a in enumerate(a_range):
            cf_scaled_value = analyze_single_point(sigma, d, a, lam, qfunc)
            CF_scaled_grid[i, j] = cf_scaled_value
            
            current_point += 1
            if current_point % (total_points // 10) == 0:
                progress = 100 * current_point / total_points
                print(f"    Progress: {progress:.0f}%")
    
    return A_grid, D_grid, CF_scaled_grid


def create_single_contour_plot(A_grid, D_grid, CF_scaled_grid, sigma, lam, a_range, d_range):
    """
    Create a single contour plot for fixed sigma and lambda values.
    
    Args:
        A_grid, D_grid, CF_scaled_grid: 2D arrays from parameter space scan
        sigma: Field correlation parameter
        lam: Coupling strength
        a_range: Array of a values
        d_range: Array of d values
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Mask invalid values (NaN or zero)
    valid_mask = ~np.isnan(CF_scaled_grid) & (CF_scaled_grid > 0)
    CF_masked = np.where(valid_mask, CF_scaled_grid, np.nan)
    
    # Create contour plot
    if np.any(valid_mask):
        # Use logarithmic scaling for better visualization
        min_cf = np.nanmin(CF_masked)
        max_cf = np.nanmax(CF_masked)
        
        if min_cf > 0 and max_cf > min_cf:
            levels = np.logspace(np.log10(min_cf), np.log10(max_cf), 25)
            
            # Filled contour plot
            contour_filled = ax.contourf(A_grid, D_grid, CF_masked, 
                                       levels=levels, norm=LogNorm(), 
                                       cmap='viridis', alpha=0.8)
            
            # Contour lines
            contour_lines = ax.contour(A_grid, D_grid, CF_masked, 
                                     levels=levels[::4], norm=LogNorm(), 
                                     colors='white', alpha=0.7, linewidths=1.0)
            
            # Colorbar
            cbar = plt.colorbar(contour_filled, ax=ax, shrink=0.9)
            cbar.set_label('CF/λ² (Scaled Contextual Fraction)', fontsize=14)
            
            # Add contour labels
            ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.4f')
        else:
            # Fallback for cases with limited valid data
            scatter = ax.scatter(A_grid[valid_mask], D_grid[valid_mask], 
                               c=CF_masked[valid_mask], s=30, cmap='viridis')
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.9)
            cbar.set_label('CF/λ² (Scaled Contextual Fraction)', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
               ha='center', va='center', fontsize=16)
    
    # Formatting
    ax.set_xlabel('a (detector size)', fontsize=14)
    ax.set_ylabel('d (detector separation)', fontsize=14)
    ax.set_title(f'Scaled Contextual Fraction (CF/λ²) vs (a,d)\nσ = {sigma:.1f}, λ = {lam:.3f}', 
                fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Add ratio lines for reference
    if len(a_range) > 1 and len(d_range) > 1:
        a_min, a_max = a_range[0], a_range[-1]
        d_min, d_max = d_range[0], d_range[-1]
        
        # a/d = 1 line
        if d_min <= a_max and a_min <= d_max:
            line_d = np.linspace(max(d_min, a_min), min(d_max, a_max), 100)
            line_a = line_d
            ax.plot(line_a, line_d, 'r--', alpha=0.8, linewidth=3, 
                   label='a/d = 1')
        
        # a/d = 0.5 and a/d = 2 lines
        if d_min <= 2*a_max and 0.5*a_min <= d_max:
            line_d = np.linspace(max(d_min, 0.5*a_min), min(d_max, 2*a_max), 100)
            line_a = 0.5 * line_d
            mask = (line_a >= a_min) & (line_a <= a_max)
            if np.any(mask):
                ax.plot(line_a[mask], line_d[mask], 'orange', linestyle=':', 
                       alpha=0.8, linewidth=2.5, label='a/d = 0.5')
        
        if 2*d_min <= a_max and a_min <= 0.5*d_max:
            line_a = np.linspace(max(a_min, 2*d_min), min(a_max, 0.5*d_max), 100)
            line_d = 0.5 * line_a
            mask = (line_d >= d_min) & (line_d <= d_max)
            if np.any(mask):
                ax.plot(line_a[mask], line_d[mask], 'cyan', linestyle=':', 
                       alpha=0.8, linewidth=2.5, label='a/d = 2')
        
        ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('scaled_contextual_fraction_contour.png', dpi=300, 
                bbox_inches='tight')
    plt.show()


def analyze_contour_features(A_grid, D_grid, CF_scaled_grid, sigma, a_range, d_range):
    """
    Analyze and report key features of the contour plot.
    
    Args:
        A_grid, D_grid, CF_scaled_grid: 2D arrays from parameter space scan
        sigma: Sigma value used
        a_range: Array of a values
        d_range: Array of d values
    
    Returns:
        str: Markdown-formatted analysis results
    """
    markdown_output = []
    markdown_output.append("## Scaled Contextual Fraction Analysis Summary\n")
    markdown_output.append(f"**Analysis for σ = {sigma:.1f}**\n")
    
    # Basic statistics
    valid_mask = ~np.isnan(CF_scaled_grid) & (CF_scaled_grid > 0)
    valid_cf_scaled = CF_scaled_grid[valid_mask]
    
    if len(valid_cf_scaled) > 0:
        max_cf_scaled = np.max(valid_cf_scaled)
        min_cf_scaled = np.min(valid_cf_scaled)
        mean_cf_scaled = np.mean(valid_cf_scaled)
        
        # Find maximum location
        max_idx = np.unravel_index(np.argmax(CF_scaled_grid), CF_scaled_grid.shape)
        max_a = A_grid[max_idx]
        max_d = D_grid[max_idx]
        
        markdown_output.append("### Statistical Summary\n")
        markdown_output.append(f"- **Valid points**: {len(valid_cf_scaled)}/{CF_scaled_grid.size} " +
                              f"({100*len(valid_cf_scaled)/CF_scaled_grid.size:.1f}%)\n")
        markdown_output.append(f"- **CF/λ² range**: [{min_cf_scaled:.6f}, {max_cf_scaled:.6f}]\n")
        markdown_output.append(f"- **Mean CF/λ²**: {mean_cf_scaled:.6f}\n")
        markdown_output.append(f"- **Maximum at**: a = {max_a:.3f}, d = {max_d:.3f}\n")
        markdown_output.append(f"- **a/d ratio at max**: {max_a/max_d:.3f}\n\n")
        
        # Analyze along a/d = 1 line
        diagonal_cf_scaled = []
        diagonal_coords = []
        
        for i in range(len(a_range)):
            for j in range(len(d_range)):
                if abs(a_range[i] - d_range[j]) < 0.2:  # Near diagonal
                    if valid_mask[j, i]:
                        diagonal_cf_scaled.append(CF_scaled_grid[j, i])
                        diagonal_coords.append(a_range[i])
        
        if len(diagonal_cf_scaled) > 0:
            max_diag_cf_scaled = np.max(diagonal_cf_scaled)
            max_diag_idx = np.argmax(diagonal_cf_scaled)
            max_diag_coord = diagonal_coords[max_diag_idx]
            markdown_output.append("### Diagonal Analysis (a/d ≈ 1)\n")
            markdown_output.append(f"- **Max CF/λ² on a/d≈1 line**: {max_diag_cf_scaled:.6f} " +
                                  f"at a≈d≈{max_diag_coord:.3f}\n\n")
    else:
        markdown_output.append("### Statistical Summary\n")
        markdown_output.append("- **No valid CF/λ² values found**\n\n")
    
    # Print to console as well
    print("\nScaled Contextual Fraction Analysis Summary:")
    print("=" * 50)
    print(f"σ = {sigma:.1f}:")
    
    if len(valid_cf_scaled) > 0:
        print(f"  Valid points: {len(valid_cf_scaled)}/{CF_scaled_grid.size} " +
              f"({100*len(valid_cf_scaled)/CF_scaled_grid.size:.1f}%)")
        print(f"  CF/λ² range: [{min_cf_scaled:.6f}, {max_cf_scaled:.6f}]")
        print(f"  Mean CF/λ²: {mean_cf_scaled:.6f}")
        print(f"  Maximum at: a = {max_a:.3f}, d = {max_d:.3f}")
        print(f"  a/d ratio at max: {max_a/max_d:.3f}")
        
        if len(diagonal_cf_scaled) > 0:
            print(f"  Max CF/λ² on a/d≈1 line: {max_diag_cf_scaled:.6f} " +
                  f"at a≈d≈{max_diag_coord:.3f}")
    else:
        print("  No valid CF/λ² values found")
    
    return "".join(markdown_output)


def write_markdown_report(sigma, lam, a_range, d_range, analysis_results, 
                         valid_count, total_count):
    """
    Write a comprehensive markdown report of the analysis.
    
    Args:
        sigma, lam: Analysis parameters
        a_range, d_range: Parameter ranges used
        analysis_results: Results from analyze_contour_features
        valid_count, total_count: Statistics from the analysis
    """
    from datetime import datetime
    
    filename = f"scaled_contextual_fraction_analysis_sigma{sigma:.1f}_lambda{lam:.3f}.md"
    
    with open(filename, 'w') as f:
        # Header
        f.write("# Scaled Contextual Fraction Contour Analysis Report\n\n")
        f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Analysis parameters
        f.write("## Analysis Parameters\n\n")
        f.write(f"- **Coupling strength (λ)**: {lam:.3f}\n")
        f.write(f"- **Field correlation (σ)**: {sigma:.1f}\n")
        f.write(f"- **Detector size range (a)**: [{a_range[0]:.1f}, {a_range[-1]:.1f}] ({len(a_range)} points)\n")
        f.write(f"- **Detector separation range (d)**: [{d_range[0]:.1f}, {d_range[-1]:.1f}] ({len(d_range)} points)\n")
        f.write("- **Q-function**: QregHeavsde\n")
        f.write(f"- **Total calculations**: {len(a_range) * len(d_range)}\n")
        f.write("- **Output quantity**: Scaled contextual fraction (CF/λ²)\n\n")
        
        # Calculation statistics
        f.write("## Calculation Statistics\n\n")
        f.write(f"- **Valid results**: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)\n")
        f.write(f"- **Failed calculations**: {total_count - valid_count} ({100*(total_count - valid_count)/total_count:.1f}%)\n\n")
        
        # Physical validity check
        f.write("## Physical Validity\n\n")
        d_sigma_ratio = d_range[0] / sigma
        if d_sigma_ratio > 2:
            f.write(f"✅ **Physical condition satisfied**: d_min/σ = {d_sigma_ratio:.2f} > 2\n\n")
        else:
            f.write(f"⚠️ **Physical condition**: d_min/σ = {d_sigma_ratio:.2f} ≤ 2 (marginal)\n\n")
        
        # Analysis results
        f.write(analysis_results)
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Scaling Rationale\n")
        f.write("The contextual fraction is scaled by λ² to remove the direct dependence on coupling strength:\n\n")
        f.write("```\nCF_scaled = CF / λ²\n```\n\n")
        f.write("This scaling reveals the fundamental parameter dependence independent of the coupling regime.\n\n")
        
        f.write("### Parameter Space Exploration\n")
        f.write("- **Grid scan**: Systematic exploration of (a,d) parameter combinations\n")
        f.write("- **Density matrix generation**: Using QregHeavsde Q-function\n")
        f.write("- **Contextual fraction calculation**: Linear programming optimization\n")
        f.write("- **Validation**: Density matrix trace and positivity checks\n\n")
        
        # Outputs
        f.write("## Generated Outputs\n\n")
        f.write("1. **Contour plot**: `scaled_contextual_fraction_contour.png`\n")
        f.write("2. **Analysis report**: This markdown file\n\n")
        
        # Interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("### Contour Plot Features\n")
        f.write("- **Color intensity**: Higher values indicate larger CF/λ²\n")
        f.write("- **Contour lines**: Lines of constant CF/λ² values\n")
        f.write("- **Reference lines**: \n")
        f.write("  - Red dashed: a/d = 1 (equal detector parameters)\n")
        f.write("  - Orange dotted: a/d = 0.5\n")
        f.write("  - Cyan dotted: a/d = 2\n\n")
        
        f.write("### Physical Significance\n")
        f.write("- **Maximum regions**: Optimal detector configurations for quantum field harvesting\n")
        f.write("- **Parameter sensitivity**: Gradient steepness indicates sensitivity to parameter changes\n")
        f.write("- **Scaling behavior**: CF/λ² reveals fundamental physics beyond coupling effects\n\n")
    
    return filename


def main():
    """
    Main function for single scaled contextual fraction contour analysis.
    """
    print("Scaled Contextual Fraction Contour Analysis: CF/λ² vs (a,d)")
    print("=" * 65)
    print("Fixed σ = 1.4 and λ = 0.02")
    print()
    
    # Fixed parameters
    lam = 0.02  # Coupling strength
    sigma = 1.4  # Field correlation parameter
    
    # Parameter ranges for a and d
    # Choose ranges that ensure d >> sigma for good physics
    a_range = np.linspace(1.0, 15.0, 30)  # Detector size parameter
    d_range = np.linspace(3.0, 25.0, 30)  # Detector separation
    
    print("Analysis parameters:")
    print(f"  λ = {lam:.3f}")
    print(f"  σ = {sigma:.1f}")
    print(f"  a range: [{a_range[0]:.1f}, {a_range[-1]:.1f}] " +
          f"({len(a_range)} points)")
    print(f"  d range: [{d_range[0]:.1f}, {d_range[-1]:.1f}] " +
          f"({len(d_range)} points)")
    print("  Q-function: QregHeavsde")
    print(f"  Total calculations: {len(a_range) * len(d_range)}")
    print("  Output: Scaled contextual fraction (CF/λ²)")
    print()
    
    # Check that d >> sigma for physical validity
    if d_range[0] <= 2 * sigma:
        print(f"Warning: Minimum d ({d_range[0]:.1f}) not much larger than σ ({sigma:.1f})")
        print("Consider using larger d values for better physics")
        print()
    
    # Perform analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress optimization warnings
        
        print(f"Analyzing σ = {sigma:.1f}, λ = {lam:.3f}...")
        
        # Scan the (a,d) parameter space
        A_grid, D_grid, CF_scaled_grid = scan_a_d_space(sigma, lam, a_range, d_range)
        
        # Quick statistics
        valid_count = np.sum(~np.isnan(CF_scaled_grid) & (CF_scaled_grid > 0))
        total_count = CF_scaled_grid.size
        print(f"Valid results: {valid_count}/{total_count} " +
              f"({100*valid_count/total_count:.1f}%)")
        print()
    
    # Create contour plot
    print("Creating scaled contextual fraction contour plot...")
    create_single_contour_plot(A_grid, D_grid, CF_scaled_grid, sigma, lam,
                              a_range, d_range)
    
    # Analyze contour features and get markdown results
    print("Analyzing contour features...")
    analysis_results = analyze_contour_features(A_grid, D_grid, CF_scaled_grid, 
                                               sigma, a_range, d_range)
    
    # Write markdown report
    print("Writing markdown report...")
    markdown_filename = write_markdown_report(sigma, lam, a_range, d_range, 
                                            analysis_results, valid_count, total_count)
    
    # Summary insights
    print("\nPhysical Insights:")
    print("-" * 25)
    print(f"• Fixed coupling strength λ = {lam:.3f}")
    print(f"• Fixed field correlation σ = {sigma:.1f}")
    print("• Contour plot shows CF/λ² landscape in (a,d) parameter space")
    print("• Scaling by λ² removes coupling strength dependence")
    print("• Reference lines show constant a/d ratios")
    print("• Look for optimal regions where CF/λ² is maximized")
    print("• Maximum CF/λ² location indicates best detector configuration")
    
    print("\nGenerated Files:")
    print(f"• Visualization: 'scaled_contextual_fraction_contour.png'")
    print(f"• Analysis report: '{markdown_filename}'")
    print("Scaled contextual fraction analysis complete!")


if __name__ == "__main__":
    main()
