import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

def plot_level1_comparison(vegas_result, nis_result, function, true_value):
    """
    Creates a 3-panel publication quality matplotlib figure comparing VEGAS and NIS:
    Panel 1: The Function & Sample Distribution
    Panel 2: Convergence
    Panel 3: Variance Reduction
    """
    # Setup aesthetic base
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        # Fallback for older environments
        plt.style.use('seaborn-paper')
    except Exception:
        pass
    
    # Custom colors and fonts
    c_vegas = '#e63946'
    c_nis = '#457b9d'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif']
    
    # Helper to extract data transparently from dicts or objects
    def get_data(res, key):
        if isinstance(res, dict):
            return res.get(key)
        return getattr(res, key, None)
        
    v_samples = get_data(vegas_result, 'samples')
    n_samples = get_data(nis_result, 'samples')
    
    v_hist = get_data(vegas_result, 'history')
    n_hist = get_data(nis_result, 'history')
    
    v_var = get_data(vegas_result, 'variance')
    n_var = get_data(nis_result, 'variance')
    
    # Estimate variance from last std + n_evals if missing
    if v_var is None and v_hist is not None:
        last_row = v_hist.iloc[-1]
        v_var = (last_row['std'] ** 2) * last_row['n_evals']
    if n_var is None and n_hist is not None:
        last_row = n_hist.iloc[-1]
        n_var = (last_row['std'] ** 2) * last_row['n_evals']

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 3, 3])
    
    # --- Panel 1: The Function ---
    ax1 = fig.add_subplot(gs[0])
    x_min, x_max = function.integration_bounds
    x = np.linspace(x_min, x_max, 1000)
    y = function(x)
    
    ax1.plot(x, y, color='black', linewidth=1.5, zorder=1)
    ax1.fill_between(x, y, color='lightgray', alpha=0.5, zorder=0)
    
    def calc_near_peak(samples):
        if samples is None or len(samples) == 0: return 0.0
        s = np.array(samples).flatten()
        near = np.sum(np.abs(s - function.center) < 2 * function.width)
        return 100.0 * near / len(s)

    v_near = calc_near_peak(v_samples)
    n_near = calc_near_peak(n_samples)
    
    if v_samples is not None:
        v_s = np.array(v_samples).flatten()
        ax1.scatter(v_s, function(v_s), label='VEGAS', color=c_vegas, marker='x', alpha=0.4, zorder=2)
    if n_samples is not None:
        n_s = np.array(n_samples).flatten()
        ax1.scatter(n_s, function(n_s), label='NIS', color=c_nis, marker='o', alpha=0.4, zorder=3, edgecolors='none')
        
    ax1.set_title(f"Sample Distribution\nVEGAS: {v_near:.1f}% near peak | NIS: {n_near:.1f}% near peak", fontsize=12)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    
    # --- Panel 2: Convergence ---
    ax2 = fig.add_subplot(gs[1])
    if v_hist is not None and n_hist is not None:
        def plot_conv(ax, hist, color, label):
            ns = hist['n_evals'].values
            mean = hist['mean'].values
            std = hist['std'].values
            err = np.abs(mean - true_value)
            
            ax.plot(ns, err, color=color, label=label, marker='o', markersize=4)
            # Shade error bands (avoid negative for log scale)
            lower = np.maximum(err - std, 1e-15)
            upper = err + std
            ax.fill_between(ns, lower, upper, color=color, alpha=0.2)
            
        plot_conv(ax2, v_hist, c_vegas, 'VEGAS')
        plot_conv(ax2, n_hist, c_nis, 'NIS')
        
        # Reference 1/sqrt(N) line
        ns = v_hist['n_evals'].values
        if len(ns) > 0:
            start_err = np.abs(v_hist['mean'].values[0] - true_value)
            if start_err == 0: start_err = v_hist['std'].values[0]
            if start_err == 0: start_err = 1.0
            ref_line = start_err * np.sqrt(ns[0]) / np.sqrt(ns)
            ax2.plot(ns, ref_line, '--', color='gray', label=r'$\propto 1/\sqrt{N}$')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel("n_evals")
        ax2.set_ylabel("|Error|")
        ax2.set_title("Convergence", fontsize=12)
        ax2.legend()
        
        if v_var and n_var and n_var > 0:
            var_red = v_var / n_var
            # Annotate at rightmost point
            ax2.annotate(f"{var_red:.1f}x Var Red.", 
                         xy=(ns[-1], np.abs(n_hist['mean'].values[-1] - true_value)),
                         xytext=(10, 0), textcoords='offset points', color=c_nis, va='center', fontsize=10)

    # --- Panel 3: Variance Reduction ---
    ax3 = fig.add_subplot(gs[2])
    if v_var is not None and n_var is not None:
        labels = ['VEGAS', 'NIS']
        variances = [v_var, n_var]
        colors = [c_vegas, c_nis]
        
        bars = ax3.barh(labels, variances, color=colors, height=0.5)
        ax3.set_xscale('log')
        ax3.set_title("Variance Reduction", fontsize=12)
        ax3.set_xlabel("Variance (log scale)")
        
        if n_var > 0:
            ratio = v_var / n_var
            ax3.text(0.5, 0.5, f"NIS is {ratio:.1f}x\nmore efficient", 
                     transform=ax3.transAxes, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    # Clean styling
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.suptitle("NIS vs VEGAS — Breit-Wigner 1D Resonance", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)  # Leave room for suptitle
    
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/level1_comparison.png', dpi=300, bbox_inches='tight')
    
    return fig
