# 2_qudit_contextual_fraction

This Python project is for calculating the contextual fraction of 2-qutrit states with respect to Heisenberg-Weyl operators.

> Use the notebook `src/notebooks/contextual_fraction.ipynb` for a detailed overview of how the contextual fraction of a state can be calculated using a linear program, including explanations of the relevant functions in the source code corresponding to each step. 

# 2-Qutrit Contextual Fraction Project Structure

This document provides a comprehensive overview of the project file structure and the purpose of each component.

## 📁 Project Overview

```
2_qudit_contextual_fraction/
├── 📂 src/                          # Source code modules
│   ├── __init__.py                  # Python package initialization
│   ├── 📂 utils/                    # Utility modules
│   │   ├── __init__.py
│   │   ├── operators.py             # Heisenberg-Weyl operators
│   │   ├── contexts.py              # Measurement contexts (40 contexts)
│   │   ├── commutators.py           # Commutator checking functions
│   │   ├── measurements.py          # Projectors & empirical models
│   │   ├── states.py                # Quantum state creation & analysis
│   │   └── ternary.py               # Base-3 number conversion
│   └── 📂 optimization/             # Linear programming optimization
│       ├── __init__.py
│       ├── incidence_matrix.py      # Global assignment constraint matrix
│       └── lin_prog.py              # Contextual fraction calculation
├── 📂 notebooks/                    # Jupyter notebooks
│   ├── contextual_fraction.ipynb    # Main analysis notebook
│   ├── contextual_fraction.py       # Jupytext paired Python file
├── 📂 .vscode/                      # VS Code configuration
│   ├── settings.json                # Project settings
│   ├── tasks.json                   # Build tasks (Jupytext sync)
│   └── keybindings.json             # Custom keyboard shortcuts
├── main.py                          # Main execution script
├── example.py                       # Simple usage examples
├── README.md                        # Project documentation
├── .gitignore                       # Git ignore rules
└── .git/                            # Git repository data
```

## 🔧 Core Components

### **Source Code (`src/`)**

#### **Utils Module (`src/utils/`)**
| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `operators.py` | Heisenberg-Weyl operator generation | `pauli(A)` - generates operators from symplectic vectors |
| `contexts.py` | Measurement context definitions | `C` - 40×2×4 array of context generators, `A`, `B` - backward compatibility |
| `commutators.py` | Commutator verification | `commute_check(A, B)`, `check_context_commutators()` |
| `measurements.py` | Measurement projectors & statistics | `projector(c, a, b)`, `empirical_model(rho)` |
| `states.py` | Quantum state creation & analysis | `create_*_state()`, `print_state_info()`, `get_default_test_states()` |
| `ternary.py` | Base-3 number system utilities | `to_ternary(n)` - converts numbers to 4-digit ternary |

#### **Optimization Module (`src/optimization/`)**
| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `incidence_matrix.py` | Constraint matrix for linear program | `M` - 360×81 sparse matrix mapping global assignments |
| `lin_prog.py` | Contextual fraction calculation | `contextual_fraction(rho)` - main optimization function |

### **Notebooks (`notebooks/`)**
| File | Purpose | Features |
|------|---------|----------|
| `contextual_fraction.ipynb` | Main analysis notebook | Complete theory + examples + calculations |
| `contextual_fraction.py` | Jupytext paired file | Same content as `.ipynb` in percent format |

### **Execution Scripts**
| File | Purpose | Usage |
|------|---------|-------|
| `main.py` | Comprehensive analysis | `python main.py` - analyzes multiple quantum states |
| `example.py` | Simple usage examples | `python example.py` - basic demonstrations |


## 🧮 Mathematical Framework

### **Key Dimensions**
- **Qutrit dimension**: 3 (three-level quantum system)
- **Two-qutrit system**: 3² = 9 dimensional Hilbert space
- **Symplectic vectors**: 4-dimensional (p₁,q₁,p₂,q₂) ∈ ℤ₃⁴
- **Contexts**: 40 maximal commuting sets
- **Outcomes per context**: 9 (3×3 measurements)
- **Global assignments**: 3⁴ = 81 possible deterministic assignments

### **Core Data Structures**
```python
# Context generators (40 contexts, 2 generators each, 4-component vectors)
C[40, 2, 4]  # C[i,0] = A[i], C[i,1] = B[i]

# Empirical model (40 contexts × 9 outcomes each)
E[360]  # E[9c + (3a + b)] = P(outcome (a,b) in context c)

# Incidence matrix (360 constraints × 81 global assignments)
M[360, 81]  # M[row, col] = 1 if assignment col satisfies constraint row
```

## 🎯 Usage Patterns

### **For Research**
1. **Theory and Analysis**: Use `notebooks/contextual_fraction.ipynb` for detailed exploration
2. **Quick Examples**: Run `example.py`
3. **Batch Processing**: Use `main.py` for analyzing multiple states

### **For Development**
For extending this project to calculate the contextual fraction of $n$-qudit states
1. **New Measurements**: Extend `measurements.py` 
2. **New Contexts**: Modify `contexts.py`
3. **Alternative Optimizers**: Extend `lin_prog.py`


## 🔧 Development Tools

### **VS Code Integration** (for SS)
- **Jupytext Sync**: Automatic `.py` ↔ `.ipynb` synchronization
- **Custom Tasks**: (WIP!) Quick access to common operations
- **Python Environment**: Integrated virtual environment support

### **Quality Assurance**
- **Type Checking**: NumPy array shapes documented
- **Error Handling**: Comprehensive exception handling in optimization
- **Validation**: Commutator checking ensures mathematical consistency

## 📊 Project Metrics

- **Lines of Code**: ~1000+ lines across all modules
- **Test Coverage**: Context validation, state generation, optimization
- **Dependencies**: NumPy, SciPy, IPython, Matplotlib (for notebooks)
- **Performance**: Optimized sparse matrix operations for large-scale problems

---

*This project implements research in quantum contextuality for two-qutrit systems, combining theoretical quantum information with computational optimization techniques.*
