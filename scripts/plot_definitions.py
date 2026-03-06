"""
Plot definitions for the paper "Contextuality from the Vacuum".

Each public function generates one figure from the paper.  They share a
common signature:

    def plot_xxx(output_dir: str, *, use_latex: bool = False,
                 progress_callback=None) -> str:
        ...
        return path_to_saved_pdf

The *progress_callback*, when given, is called as
``progress_callback(current_step, total_steps)`` so that callers can
display a progress bar or percentage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure ``src/`` is importable regardless of working directory
# ---------------------------------------------------------------------------
for _p in [Path(__file__).resolve().parent.parent, Path.cwd(), *Path.cwd().parents]:
    _cand = _p / "src"
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from qft.udw_qutrits import detector_state
from optimization.lin_prog import contextual_fraction
from magic.wigner_polytope import wigner_inequalities
from utils.state_checks import is_valid_state, validate_and_print

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ProgressCB = Callable[[int, int], None] | None

COLORS_MAIN = [
    "#000000", "#332288", "#117733", "#88CCEE",
    "#DDCC77", "#CC6677", "#AA4499", "#882255",
]


def _configure_matplotlib(use_latex: bool) -> None:
    """Set matplotlib rc params.  Falls back to mathtext when LaTeX is absent."""
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{lmodern}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Bitstream Vera Serif",
                           "Computer Modern Roman"],
            "mathtext.fontset": "cm",
        })


def reduce_state(rho: np.ndarray) -> np.ndarray:
    """Partial trace over the second qutrit (B) of a 9x9 density matrix."""
    rho = np.asarray(rho)
    if rho.shape != (9, 9):
        raise ValueError("rho must be a 9x9 matrix for a 2-qutrit system.")
    return np.trace(rho.reshape(3, 3, 3, 3), axis1=1, axis2=3)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ===================================================================
# Plot 1 – Contextual fraction vs Omega*T  (large detectors, R/T=1)
# ===================================================================

PLOT_NAMES = [
    "cf_large",
    "cf_small",
    "wigner_large",
    "wigner_small",
    "cf_appendix",
    "wigner_appendix",
    "cf_fixed_romega",
    "wigner_fixed_romega",
]

PLOT_DESCRIPTIONS = {
    "cf_large": "Contextual fraction vs gap (R/T=1, large detectors)",
    "cf_small": "Contextual fraction vs gap (R/T=0.1, small detectors)",
    "wigner_large": "Wigner negativity vs gap (R/T=1)",
    "wigner_small": "Wigner negativity vs gap (R/T=0.1, small detectors)",
    "cf_appendix": "CF, SU(2) vs HW (R/T=0.1, appendix)",
    "wigner_appendix": "Wigner negativity, SU(2) vs HW (R/T=0.1, appendix)",
    "cf_fixed_romega": "CF vs gap, fixed RΩ=0.01, dΩ=20 (appendix C)",
    "wigner_fixed_romega": "Wigner negativity vs gap, fixed RΩ=0.01, dΩ=20 (appendix C)",
}


def plot_cf_large(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Contextual fraction vs Omega*T for large detectors (R/T = 1)."""
    _configure_matplotlib(use_latex)

    switching = 1.0
    smearing = 1
    lam = 1e-3
    regulator = 1.0
    regularization = "heaviside"
    group_types = [("SU2", "smeared")]
    gaps = np.linspace(0.0, 4, 250)
    deez = [5, 7, 10, 14]
    colors = COLORS_MAIN
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]

    # Total work: 2 product sweeps + len(deez) entangled sweeps, each over len(gaps)
    total_sweeps = 2 + len(deez)
    total_steps = total_sweeps * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # --- product state rho_a x rho_b ---
    for group, detector_type in group_types:
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, rho1))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes \hat{\rho}_b$",
                linestyle="-", color=colors[1])

    # --- product state rho_a x |0><0| ---
    for group, detector_type in group_types:
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, (1 / 3) * np.eye(3)))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes |0\rangle\langle 0|$",
                linestyle="-", color=colors[0])

    # --- entangled states at various separations ---
    for idx, d in enumerate(deez):
        for group, detector_type in group_types:
            cf_vals = []
            for gap in gaps:
                rho = detector_state(
                    gap=gap, switching=switching,
                    separation=2 * smearing + d / np.sqrt(2),
                    regulator=regulator, smearing=smearing,
                    regularization=regularization,
                    detector_type=detector_type, group=group, lam=lam,
                )
                if is_valid_state(rho):
                    res = contextual_fraction(rho)
                    cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
                else:
                    cf_vals.append(np.nan)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)
            cf_vals = np.array(cf_vals) / lam**2
            label = rf"$\hat{{\rho}}_{{ab}},\ d=2R+{d}T/\sqrt{{2}}$"
            ax.plot(gaps, cf_vals, lw=1.5, label=label,
                    linestyle=linestyles[idx % len(linestyles)],
                    color=colors[idx + 2])

    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"CF/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    ax.legend()
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "contextual_fraction_vs_gap_large_detectors.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 2 – Contextual fraction vs Omega*T  (small detectors, R/T=0.1)
# ===================================================================

def plot_cf_small(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Contextual fraction vs Omega*T for small detectors (R/T = 0.1)."""
    _configure_matplotlib(use_latex)

    switching = 1.0
    smearing = 0.1
    lam = 1e-3
    regulator = 1.0
    regularization = "heaviside"
    group_types = [("SU2", "smeared")]
    gaps = np.linspace(0.0, 4, 250)
    deez = [5, 10]
    colors = COLORS_MAIN
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]

    total_sweeps = 2 + len(deez)
    total_steps = total_sweeps * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # --- product state rho_a x rho_b ---
    for group, detector_type in group_types:
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, rho1))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes \hat{\rho}_b$",
                linestyle="-", color=colors[1])

    # --- product state rho_a x |0><0| ---
    for group, detector_type in group_types:
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, (1 / 3) * np.eye(3)))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes |0\rangle\langle 0|$",
                linestyle="-", color=colors[0])

    # --- entangled states ---
    for idx, d in enumerate(deez):
        for group, detector_type in group_types:
            cf_vals = []
            for gap in gaps:
                rho = detector_state(
                    gap=gap, switching=switching,
                    separation=2 * smearing + d / np.sqrt(2),
                    regulator=regulator, smearing=smearing,
                    regularization=regularization,
                    detector_type=detector_type, group=group, lam=lam,
                )
                if is_valid_state(rho):
                    res = contextual_fraction(rho)
                    cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
                else:
                    cf_vals.append(np.nan)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)
            cf_vals = np.array(cf_vals) / lam**2
            label = rf"$\hat{{\rho}}_{{ab}}$, $d=2R+{d}T/\sqrt{{2}}$"
            ax.plot(gaps, cf_vals, lw=1.5, label=label,
                    linestyle=linestyles[idx % len(linestyles)],
                    color=colors[2 * idx + 1])

    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"CF/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    ax.legend()
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "contextual_fraction_vs_gap_small_detectors.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 3 – Wigner negativity vs Omega*T  (large detectors, R/T=1)
# ===================================================================

def plot_wigner_large(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Wigner negativity vs Omega*T for large SU(2) detectors (R/T = 1)."""
    _configure_matplotlib(use_latex)
    from matplotlib.ticker import MultipleLocator

    switching = 1.0
    separation = 1.0
    smearing = 1.0
    lam = 1e-3
    regulator = 1
    regularization = "magical"
    group_types = [("SU2", "smeared")]
    gaps = np.linspace(0.0, 4, 250)
    colors = COLORS_MAIN

    total_steps = len(group_types) * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group, detector_type in group_types:
        wn_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=separation,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho_1 = reduce_state(rho)
                _, _, violation_sum = wigner_inequalities(rho_1, include_sum=True)
                wn_vals.append(violation_sum * (1 / 3) * (1 / lam**2))
            else:
                wn_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        wn_vals = np.array(wn_vals)
        ax.plot(gaps[:len(wn_vals)], wn_vals, lw=1.5, color=colors[1])

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"N/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "magic_vs_gap_big_detectors.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 4 – Wigner negativity, SU(2) vs HW  (appendix, R/T=0.1)
# ===================================================================

def plot_wigner_appendix(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Wigner negativity comparing SU(2) and HW models (R/T = 0.1, appendix)."""
    _configure_matplotlib(use_latex)
    from matplotlib.ticker import MultipleLocator

    switching = 1.0
    separation = 1.0
    smearing = 0.1
    lam = 1e-3
    regulator = 1
    regularization = "magical"
    group_types = [("SU2", "smeared"), ("HW", "smeared")]
    labels = {("SU2", "smeared"): "SU(2)", ("HW", "smeared"): "HW"}
    colors = ["#E1BE6A", "#40B0A6"]
    gaps = np.linspace(0.0, 10, 250)

    total_steps = len(group_types) * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group, detector_type in group_types:
        wn_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=separation,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho_1 = reduce_state(rho)
                _, _, violation_sum = wigner_inequalities(rho_1, include_sum=True)
                wn_vals.append(violation_sum * (1 / 3) * (1 / lam**2))
            else:
                wn_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        wn_vals = np.array(wn_vals)
        ax.plot(
            gaps[:len(wn_vals)], wn_vals, lw=1.5,
            label=labels[(group, detector_type)],
            color=colors[0] if group == "SU2" else colors[1],
            linestyle="--" if group == "HW" else "-",
        )

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"N/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    ax.legend()
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "magic_vs_gap_different_models.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 5 – Wigner negativity vs Omega*T  (small detectors, R/T=0.1)
# Figure 1(d) in the paper
# ===================================================================

def plot_wigner_small(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Wigner negativity vs Omega*T for small SU(2) detectors (R/T = 0.1)."""
    _configure_matplotlib(use_latex)
    from matplotlib.ticker import MultipleLocator

    switching = 1.0
    separation = 1.0
    smearing = 0.1
    lam = 1e-3
    regulator = 1
    regularization = "magical"
    group_types = [("SU2", "smeared")]
    gaps = np.linspace(0.0, 4, 250)
    colors = COLORS_MAIN

    total_steps = len(group_types) * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group, detector_type in group_types:
        wn_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=separation,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho_1 = reduce_state(rho)
                _, _, violation_sum = wigner_inequalities(rho_1, include_sum=True)
                wn_vals.append(violation_sum * (1 / 3) * (1 / lam**2))
            else:
                wn_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        wn_vals = np.array(wn_vals)
        ax.plot(gaps[:len(wn_vals)], wn_vals, lw=1.5, color=colors[1])

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"N/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "magic_vs_gap_small_detectors.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 6 – Contextual fraction, SU(2) vs HW  (appendix, R/T=0.1)
# Figure 2(a) in the paper
# ===================================================================

def plot_cf_appendix(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Contextual fraction comparing SU(2) and HW models (R/T = 0.1, appendix)."""
    _configure_matplotlib(use_latex)

    switching = 1.0
    smearing = 0.1
    lam = 1e-3
    regulator = 1.0
    regularization = "heaviside"
    group_types = [("SU2", "smeared"), ("HW", "smeared")]
    labels = {("SU2", "smeared"): "SU(2)", ("HW", "smeared"): "HW"}
    colors = ["#E1BE6A", "#40B0A6"]
    gaps = np.linspace(0.0, 10, 250)
    deez = [5, 10]
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]

    # 2 product sweeps + len(deez) entangled sweeps, each for both groups
    total_sweeps = (2 + len(deez)) * len(group_types)
    total_steps = total_sweeps * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group_idx, (group, detector_type) in enumerate(group_types):
        color = colors[0] if group == "SU2" else colors[1]
        ls_solid = "-" if group == "SU2" else "--"

        # --- product state rho_a x rho_b ---
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, rho1))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        lbl = rf"$\hat{{\rho}}_a \otimes \hat{{\rho}}_b$, {labels[(group, detector_type)]}"
        ax.plot(gaps, cf_vals, lw=1.5, label=lbl, linestyle=ls_solid, color=color)

        # --- product state rho_a x |0><0| ---
        cf_vals = []
        for gap in gaps:
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, (1 / 3) * np.eye(3)))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        lbl = rf"$\hat{{\rho}}_a \otimes |0\rangle\langle 0|$, {labels[(group, detector_type)]}"
        ax.plot(gaps, cf_vals, lw=1.5, label=lbl,
                linestyle=(0, (1, 1)) if group == "SU2" else (0, (5, 1)),
                color=color)

        # --- entangled states ---
        for idx, d in enumerate(deez):
            cf_vals = []
            for gap in gaps:
                rho = detector_state(
                    gap=gap, switching=switching,
                    separation=2 * smearing + d / np.sqrt(2),
                    regulator=regulator, smearing=smearing,
                    regularization=regularization,
                    detector_type=detector_type, group=group, lam=lam,
                )
                if is_valid_state(rho):
                    res = contextual_fraction(rho)
                    cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
                else:
                    cf_vals.append(np.nan)
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)
            cf_vals = np.array(cf_vals) / lam**2
            lbl = rf"$\hat{{\rho}}_{{ab}}$, $d=2R+{d}T/\sqrt{{2}}$, {labels[(group, detector_type)]}"
            ax.plot(gaps, cf_vals, lw=1.5, label=lbl,
                    linestyle=linestyles[idx % len(linestyles)],
                    color=color)

    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"CF/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R/T={smearing}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    ax.legend(fontsize=7)
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "contextual_fraction_su2_vs_hw.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 7 – Contextual fraction, fixed R*Omega and d*Omega  (appendix C)
# Figure 3(a) in the paper
# ===================================================================

def plot_cf_fixed_romega(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Contextual fraction vs Omega*T with fixed R*Omega=0.01 and d*Omega=20 (appendix C)."""
    _configure_matplotlib(use_latex)

    switching = 1.0
    lam = 1e-3
    regulator = 1.0
    regularization = "heaviside"
    group_types = [("SU2", "smeared")]
    R_omega = 0.01
    d_omega = 20.0
    # Avoid gap=0 (would give R→∞); start from a small positive value.
    gaps = np.linspace(0.01, 4, 250)
    colors = COLORS_MAIN
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1))]

    # 2 product sweeps + 1 entangled sweep (single fixed dΩ)
    total_sweeps = 2 + 1
    total_steps = total_sweeps * len(gaps) * len(group_types)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group, detector_type in group_types:

        # --- product state rho_a x rho_b ---
        cf_vals = []
        for gap in gaps:
            smearing = R_omega / gap
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, rho1))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes \hat{\rho}_b$",
                linestyle="-", color=colors[1])

        # --- product state rho_a x |0><0| ---
        cf_vals = []
        for gap in gaps:
            smearing = R_omega / gap
            rho = detector_state(
                gap=gap, switching=switching, separation=0,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho1 = reduce_state(rho)
                res = contextual_fraction(np.kron(rho1, (1 / 3) * np.eye(3)))
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=r"$\hat{\rho}_a \otimes |0\rangle\langle 0|$",
                linestyle="-", color=colors[0])

        # --- entangled state at fixed d*Omega ---
        cf_vals = []
        for gap in gaps:
            smearing = R_omega / gap
            separation = d_omega / gap
            rho = detector_state(
                gap=gap, switching=switching, separation=separation,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                res = contextual_fraction(rho)
                cf_vals.append(res.get("b", np.nan) if res.get("success") else np.nan)
            else:
                cf_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        cf_vals = np.array(cf_vals) / lam**2
        ax.plot(gaps, cf_vals, lw=1.5,
                label=rf"$\hat{{\rho}}_{{ab}}$, $R\Omega={R_omega}$, $d\Omega={d_omega:.0f}$",
                linestyle=linestyles[0], color=colors[2])

    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"CF/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R\Omega={R_omega}$, $d\Omega={d_omega:.0f}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    ax.legend()
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "contextual_fraction_fixed_romega.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Plot 8 – Wigner negativity, fixed R*Omega and d*Omega  (appendix C)
# Figure 3(b) in the paper
# ===================================================================

def plot_wigner_fixed_romega(
    output_dir: str,
    *,
    use_latex: bool = False,
    progress_callback: ProgressCB = None,
) -> str:
    """Wigner negativity vs Omega*T with fixed R*Omega=0.01 and d*Omega=20 (appendix C)."""
    _configure_matplotlib(use_latex)
    from matplotlib.ticker import MultipleLocator

    switching = 1.0
    lam = 1e-3
    regulator = 1
    regularization = "magical"
    group_types = [("SU2", "smeared")]
    R_omega = 0.01
    d_omega = 20.0
    # Avoid gap=0 (would give R→∞); start from a small positive value.
    gaps = np.linspace(0.01, 4, 250)
    colors = COLORS_MAIN

    total_steps = len(group_types) * len(gaps)
    step = 0

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    for group, detector_type in group_types:
        wn_vals = []
        for gap in gaps:
            smearing = R_omega / gap
            separation = d_omega / gap
            rho = detector_state(
                gap=gap, switching=switching, separation=separation,
                regulator=regulator, smearing=smearing,
                regularization=regularization,
                detector_type=detector_type, group=group, lam=lam,
            )
            if is_valid_state(rho):
                rho_1 = reduce_state(rho)
                _, _, violation_sum = wigner_inequalities(rho_1, include_sum=True)
                wn_vals.append(violation_sum * (1 / 3) * (1 / lam**2))
            else:
                wn_vals.append(np.nan)
            step += 1
            if progress_callback:
                progress_callback(step, total_steps)
        wn_vals = np.array(wn_vals)
        ax.plot(gaps[:len(wn_vals)], wn_vals, lw=1.5, color=colors[1])

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r"$\Omega T$", fontsize=14)
    ax.set_ylabel(r"N/$\lambda^2$", fontsize=14)
    ax.set_title(rf"$R\Omega={R_omega}$, $d\Omega={d_omega:.0f}$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.1)
    fig.tight_layout()

    out = _ensure_dir(output_dir) / "magic_vs_gap_fixed_romega.pdf"
    fig.savefig(str(out))
    plt.close(fig)
    return str(out)


# ===================================================================
# Registry – maps short names to (function, description)
# ===================================================================

PLOTS = {
    "cf_large":           (plot_cf_large,           PLOT_DESCRIPTIONS["cf_large"]),
    "cf_small":           (plot_cf_small,           PLOT_DESCRIPTIONS["cf_small"]),
    "wigner_large":       (plot_wigner_large,       PLOT_DESCRIPTIONS["wigner_large"]),
    "wigner_small":       (plot_wigner_small,       PLOT_DESCRIPTIONS["wigner_small"]),
    "cf_appendix":        (plot_cf_appendix,        PLOT_DESCRIPTIONS["cf_appendix"]),
    "wigner_appendix":    (plot_wigner_appendix,    PLOT_DESCRIPTIONS["wigner_appendix"]),
    "cf_fixed_romega":    (plot_cf_fixed_romega,    PLOT_DESCRIPTIONS["cf_fixed_romega"]),
    "wigner_fixed_romega":(plot_wigner_fixed_romega,PLOT_DESCRIPTIONS["wigner_fixed_romega"]),
}
