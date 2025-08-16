# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- # Main plots
#
# > This notebook contains the plots of contextual fraction vs various physical quantities for two qutrit UDW detectors interacting with a quantum field. -->
#

# %%
'''This cell imports the necessary libraries and modules for the analysis.'''
import sys, os
from pathlib import Path

# Robustly ensure 'src' is on sys.path (works whether CWD is repo root or notebooks/)
for p in [Path.cwd(), *Path.cwd().parents]:
    cand = p / "src"
    if cand.exists():
        sys.path.insert(0, str(cand))
        break

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Importing modules
from qft.udw_qutrits import detector_state      
from optimization.lin_prog import contextual_fraction
from utils.state_checks import is_valid_state, validate_and_print


plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


# %% [markdown]
# ## Contextual fraction vs gap ($\Omega$)

# %%
'''This cell generates the plots for contextual fraction vs gap for different detector types.'''


# Fixed parameters
switching = 1.0
separation = 1.0
smearing = 0.1
lam = 1e-3
regulator = 1.0
regularization = "heaviside"  # used by Q_term but detector_type/group decide which terms contribute

group_types = [("SU2", "point_like"), ("SU2", "smeared"), ("HW", "smeared")]
labels = {
    ("SU2", "point_like"): "SU(2), point like",
    ("SU2", "smeared"): "SU(2), smeared",
    ("HW", "smeared"): "HW, smeared"
}

plt.figure(figsize=(4.8,3.6))

# Sweep gaps
gaps = np.linspace(0.1, 5, 100)  # avoid zero to keep numerics well-behaved


for group, detector_type in group_types:
    cf_vals = []
    for gap in gaps:
        rho = detector_state(
            gap=gap,
            switching=switching,
            separation=separation,
            regulator=regulator,
            smearing=smearing,
            regularization=regularization,
            detector_type=detector_type,
            group=group,
            lam=lam,
        )
        if is_valid_state(rho):
            res = contextual_fraction(rho)
            if not res.get("success", False):
                cf_vals.append(np.nan)
            else:
                cf_vals.append(res.get("b", np.nan))
        else:
            print("Invalid state detected at gap =", gap)
            validate_and_print(rho)
            break

    cf_vals = np.array(cf_vals) #deleted division by lambda^2
    plt.plot(gaps, cf_vals, marker=".", lw=1.5, label=labels[(group, detector_type)])


plt.xlabel(r"{Gap} $\Omega T$",fontsize=14)
plt.ylabel(r"{Contextual fraction} CF",fontsize=14)
plt.title(rf"(a) $d/\sigma={separation}$, $R/\sigma={smearing}$, $\lambda={lam}$",fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(title="Detector group and type")
plt.tight_layout()
# plt.savefig("contextual_fraction_vs_gap_different_models.pdf")
plt.show()



# %% [markdown]
# ## Contextual fraction vs gap ($\Omega$)
# ### ONLY FOR SU2

# %%
''' This cell generates the plots for contextual fraction for various detector types only for the SU(2) group. '''

# Fixed parameters
switching = 1.0
separation = 1.0
smearing = 0.1
lam = 1e-3
regulator = 1.0
regularization = "heaviside"  # used by Q_term but detector_type/group decide which terms contribute

# You can toggle these if you want SU2 instead
group = "SU2"            # or "SU2"
detector_type = "smeared"  # "point_like" or "smeared"

plt.figure(figsize=(4.8,3.6))

# Sweep gaps
gaps = np.linspace(0.1, 20, 100)  # avoid zero to keep numerics well-behaved
cf_vals = []

for gap in gaps:
    rho = detector_state(
        gap=gap,
        switching=switching,
        separation=gap,
        regulator=regulator,
        smearing=smearing,
        regularization=regularization,
        detector_type=detector_type,
        group=group,
        lam=lam,
    )
    if is_valid_state(rho):
        res = contextual_fraction(rho)
        if not res.get("success", False):
            cf_vals.append(np.nan)

        else:
            cf_vals.append(res.get("b", np.nan))
    else:
        print("Invalid state detected at gap =", gap)
        validate_and_print(rho)
        break #Print the details for the first invalid state and exit

cf_vals = np.array(cf_vals)#*(1/lam**2)

plt.plot(gaps, cf_vals, marker=".", lw=1.5)
plt.xlabel(r"{Gap} $\Omega T$",fontsize=14)
plt.ylabel(r"{Contextual fraction} CF",fontsize=14)
plt.title(rf"(a) $d/\sigma={separation}$, $R/\sigma={smearing}$, $\lambda={lam}$",fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("contextual_fraction_vs_gap.pdf")
plt.show()

# %% [markdown]
# ## Contextual fraction vs switching (T)

# %%
'''This cell generates the plots for contextual fraction vs switching for different detector types.'''


# Fixed parameters
gap = 1.0
separation = 10
smearing = 0.01
lam = 1e-3
regulator = 1.0
regularization = "heaviside"  # used by Q_term but detector_type/group decide which terms contribute

group_types = [("SU2", "point_like"), ("SU2", "smeared"), ("HW", "smeared")]
labels = {
    ("SU2", "point_like"): "SU(2), point like",
    ("SU2", "smeared"): "SU(2), smeared",
    ("HW", "smeared"): "HW, smeared"
}

plt.figure(figsize=(4.8,3.6))

# Sweep gaps
switching_values = np.linspace(1, 10, 50)  # avoid zero to keep numerics well-behaved


for group, detector_type in group_types:
    cf_vals = []
    for switching in switching_values:
        rho = detector_state(
            gap=gap,
            switching=switching,
            separation=separation,
            regulator=regulator,
            smearing=smearing,
            regularization=regularization,
            detector_type=detector_type,
            group=group,
            lam=lam,
        )
        if is_valid_state(rho):
            res = contextual_fraction(rho)
            if not res.get("success", False):
                cf_vals.append(np.nan)
            else:
                cf_vals.append(res.get("b", np.nan))
        else:
            print("Invalid state detected at T =", switching)
            validate_and_print(rho)
            break

    cf_vals = np.array(cf_vals) #deleted division by lambda^2
    plt.plot(switching_values, cf_vals, marker=".", lw=1.5, label=labels[(group, detector_type)])


plt.xlabel(r"{Switching} $T\Omega$",fontsize=14)
plt.ylabel(r"{Contextual fraction} CF",fontsize=14)
plt.title(rf"(a) $d\Omega={separation}$, $R\Omega={smearing}$, $\lambda={lam}$",fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(title="Detector group and type")
plt.tight_layout()
# plt.savefig("contextual_fraction_vs_gap_different_models.pdf")
plt.show()



# %% [markdown]
# ## Contextual fraction vs separation ($d$)
#

# %%
'''This cell generates the plots for contextual fraction vs separation for different detector types.'''


# Fixed parameters
gap = 1.0
switching = 1.0
smearing = 0.01
lam = 1e-3
regulator = 1.0
regularization = "heaviside"  # used by Q_term but detector_type/group decide which terms contribute

group_types = [("SU2", "point_like"), ("SU2", "smeared"), ("HW", "smeared")]
labels = {
    ("SU2", "point_like"): "SU(2), point like",
    ("SU2", "smeared"): "SU(2), smeared",
    ("HW", "smeared"): "HW, smeared"
}

plt.figure(figsize=(4.8,3.6))

# Sweep gaps
separation_values = np.linspace(0.1, 5, 100)  # avoid zero to keep numerics well-behaved


for group, detector_type in group_types:
    cf_vals = []
    for separation in separation_values:
        rho = detector_state(
            gap=gap,
            switching=switching,
            separation=separation,
            regulator=regulator,
            smearing=smearing,
            regularization=regularization,
            detector_type=detector_type,
            group=group,
            lam=lam,
        )
        if is_valid_state(rho):
            res = contextual_fraction(rho)
            if not res.get("success", False):
                cf_vals.append(np.nan)
            else:
                cf_vals.append(res.get("b", np.nan))
        else:
            print("Invalid state detected at d =", separation)
            validate_and_print(rho)
            break

    cf_vals = np.array(cf_vals) #deleted division by lambda^2
    plt.plot(separation_values, cf_vals, marker=".", lw=1.5, label=labels[(group, detector_type)])


plt.xlabel(r"{Separation} $d/T$",fontsize=14)
plt.ylabel(r"{Contextual fraction} CF",fontsize=14)
plt.title(rf"(a) $d/T={separation}$, $R/T={smearing}$, $\lambda={lam}$",fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(title="Detector group and type")
plt.tight_layout()
# plt.savefig("contextual_fraction_vs_gap_different_models.pdf")
plt.show()



# %% [markdown]
# ## Contextual fraction vs smearing (R)

# %%
'''This cell generates the plots for contextual fraction vs separation for different detector types.'''


# Fixed parameters
gap = 1.0
switching = 1.0
separation = 1.0
lam = 1e-3
regulator = 1.0
regularization = "heaviside"  # used by Q_term but detector_type/group decide which terms contribute

group_types = [("SU2", "point_like"), ("SU2", "smeared"), ("HW", "smeared")]
labels = {
    ("SU2", "point_like"): "SU(2), point like",
    ("SU2", "smeared"): "SU(2), smeared",
    ("HW", "smeared"): "HW, smeared"
}

plt.figure(figsize=(4.8,3.6))

# Sweep gaps
smearing_values = np.linspace(0.1, 5, 100)  # avoid zero to keep numerics well-behaved


for group, detector_type in group_types:
    cf_vals = []
    for smearing in smearing_values:
        rho = detector_state(
            gap=gap,
            switching=switching,
            separation=separation,
            regulator=regulator,
            smearing=smearing,
            regularization=regularization,
            detector_type=detector_type,
            group=group,
            lam=lam,
        )
        if is_valid_state(rho):
            res = contextual_fraction(rho)
            if not res.get("success", False):
                cf_vals.append(np.nan)
            else:
                cf_vals.append(res.get("b", np.nan))
        else:
            print("Invalid state detected at d =", separation)
            validate_and_print(rho)
            break

    cf_vals = np.array(cf_vals) #deleted division by lambda^2
    plt.plot(smearing_values, cf_vals, marker=".", lw=1.5, label=labels[(group, detector_type)])


plt.xlabel(r"{Smearing} $R/T$",fontsize=14)
plt.ylabel(r"{Contextual fraction} CF",fontsize=14)
plt.title(rf"(a) $d/\sigma={separation}$, $R/\sigma={smearing}$, $\lambda={lam}$",fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)
plt.legend(title="Detector group and type")
plt.tight_layout()
# plt.savefig("contextual_fraction_vs_gap_different_models.pdf")
plt.show()


