# GEMINI.md — nis-mc Workspace Rules

> **Always-On rule for Antigravity.**
> Loaded automatically at the start of every agent session in this workspace.
> Defines physics context, code standards, shell conventions, and agent behavior.
> Also committed to the repo as human-readable project documentation.

---

## 🧠 Project Identity

**nis-mc** benchmarks Neural Importance Sampling (normalizing flows) against VEGAS
for Monte Carlo integration in high-energy physics.

Stack: **Python 3.11 · PyTorch float64 · VEGAS library · matplotlib · pytest · ruff**
Author: Joel Müller, PhD student, Universität Bern

---

## 🐚 Shell Detection (Critical — Read First)

**Before running ANY activation or pip command, detect the user's shell:**

```bash
echo $SHELL
```

Then activate the virtual environment with the correct script:

| Shell | Activation command |
|---|---|
| Fish (`/usr/bin/fish`) | `source .venv/bin/activate.fish` |
| bash (`/bin/bash`) | `source .venv/bin/activate` |
| zsh (`/bin/zsh`) | `source .venv/bin/activate` |

**Never use `source .venv/bin/activate` in a Fish shell session** — it will error.
Always detect first, activate second.

After activation, verify:
```bash
which python   # must point to .venv/bin/python
```

---

## 🐍 Virtual Environment Rules

- The venv lives at `.venv/` inside the repo root
- Created with: `python3.11 -m venv .venv`
- It is gitignored — never commit it
- **All pip installs happen inside the venv** — never use `sudo pip` or `--break-system-packages`
- PyTorch must be installed before `pip install -e .` due to its special index URL:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu -q
  pip install -e ".[dev]" -q
  ```
- To verify environment health: `python -c "import nis_mc; import torch; print('ok')"`

---

## ⚛️ Physics Context

### The Core Problem
`∫ |M(Φ)|² dΦ_n` where `|M|²` has **violently sharp resonance peaks**
(e.g., Z-boson, Higgs) and kinematic correlations from energy-momentum conservation.

### Why VEGAS Fails
VEGAS's separable grid `w(x) = w₁(x₁) × w₂(x₂)` cannot capture diagonal correlations.
A ridge along `x₁ + x₂ = 1` (from momentum conservation) is invisible to VEGAS — its
square cells can never align with it.

### Why NIS Works
A normalizing flow `q_θ(x)` learns the joint geometry of `|M|²`.
Reverse KL minimization: `L(θ) = E_{x~q}[-log(|M|²(x)/q_θ(x))]`
→ weights `w_i = f(x_i)/q_θ(x_i)` become flat → variance → 0.

### Physical Conventions
- Natural units: `ℏ = c = 1`
- Metric: `(+,-,-,-)` (West Coast)
- PDG 2024 values in `nis_mc/utils/constants.py` — never edit without updating this file

### PDG 2024 Constants
```python
mZ           = 91.1876   # GeV
GZ           = 2.4952    # GeV
mH           = 125.25    # GeV
sin2_theta_W = 0.23122
alpha_em     = 1/137.036
```

---

## 🏗️ Code Architecture

### Flow Stack (PyTorch throughout — no NumPy fallback)

| Level | File | Architecture |
|---|---|---|
| 1 (teaching) | `flows/affine_coupling_1d.py` | `AffineCouplingLayer1D(nn.Module)` — 2 `nn.Parameter` scalars |
| 2 (production) | `flows/realnvp.py` | `RealNVP2D` — 4 `AffineCouplingLayer2D` with MLP [1→64→64→2] |
| 3 (HEP) | `flows/realnvp.py` | `RealNVPnD` — generalised to n dims |

Training contract for all flows: **Adam · reverse KL · gradient clipping · float64**.

### Module Separation (strict)

| Module | May import | May NOT import |
|---|---|---|
| `integrators/` | `functions/`, `flows/`, `utils/` | `visualization/` |
| `functions/` | `utils/`, `numpy`, `scipy`, `torch` | `integrators/`, `flows/` |
| `flows/` | `torch`, `math` | `functions/`, `integrators/` |
| `visualization/` | `numpy`, `matplotlib`, `pandas` | `flows/` directly |
| `utils/` | `numpy`, `scipy` | `torch` (except `constants.py`) |

### BaseIntegrator Contract (never modify this interface)

```python
@dataclass
class IntegrationResult:
    mean: float
    std: float
    chi2: float
    n_evals: int
    metadata: dict   # ESS, ess_frac, training_steps, etc.

class BaseIntegrator(ABC):
    @abstractmethod
    def integrate(self, f, bounds, n_eval) -> IntegrationResult: ...
    @abstractmethod
    def benchmark(self, f, bounds, n_eval_list) -> pd.DataFrame: ...
```

Always return `IntegrationResult`, never raw tuples.

### Numerical Stability (non-negotiable)

1. `log(0)` guard: `torch.log(x.clamp(min=1e-300))`
2. Logit clamp: `x.clamp(1e-6, 1-1e-6)` before any `logit(x)`
3. Gradient clipping in all 2D+ flows: `clip_grad_norm_(params, max_norm=1.0)`
4. Seed both frameworks: `torch.manual_seed(seed)` AND `np.random.seed(seed)`
5. float64 everywhere: `dtype=torch.float64`, `.double()` on all `nn.Module`s
6. Cache integrals: `@functools.cached_property` for `true_integral`

### PyTorch Rules

- `dtype=torch.float64` for all tensors and parameters
- `self.flow.eval()` + `torch.no_grad()` during inference — always
- `.detach().numpy()` + `.clip(1e-6, 1-1e-6)` when passing to non-torch functions
- Batch sizes: 1024 (Level 1 train), 2048 (Level 2 train), 100_000 (inference)
- Optimizers: Adam lr=1e-2 (Level 1), Adam lr=1e-3 (Level 2+)

---

## 🎨 Visualization Standards

```python
plt.style.use('seaborn-v0_8-paper')

COLORS = {
    'vegas': '#e63946',  # red
    'nis':   '#457b9d',  # steel blue
    'truth': '#2d6a4f',  # green
    'ref':   '#adb5bd',  # gray (1/√N lines)
}
```

- 300 DPI for all files saved to `results/`
- Always: `ax.spines[['top', 'right']].set_visible(False)`
- Font: `DejaVu Serif`
- Every figure saved to `results/` must also be returned as a `Figure` object
- No pie charts. No default `C0`/`C1`/`C2` colors.

---

## 📝 Documentation

Every public class/function must have:
- One-line summary
- Full Args / Returns / Raises
- Physics note where relevant (e.g., "For Γ/range < 0.05, use n_eval ≥ 50_000")

**Flow math comments:** Every single line of `forward()` and `log_prob()` in any flow
must have an inline comment explaining the mathematical operation and its role.
Reference paper equations as `# Eq. (3.14) in arXiv:1912.02762` where applicable.

---

## 🛡️ Agent Safety Rules

### Always Ask Before
- Writing to `results/` (may overwrite benchmark figures)
- Installing packages not in `pyproject.toml`
- Changing a test assertion (fix the code, not the test)
- Editing `nis_mc/utils/constants.py`
- Running training loops > 1000 steps

### Never Do Without Explicit Instruction
- `sudo pip install` or `pip install --break-system-packages`
- Commit to `main` directly
- `git push --force` or `git reset --hard`
- Modify `BaseIntegrator` ABC interface
- Create directories with the agent — scaffold is done by the bootstrap prompt in Phase 2

### Destructive Command Protocol
Commands matching `rm -rf`, `git reset --hard`, `git push --force`:
→ Print `⚠️ DESTRUCTIVE: {command}` and wait for explicit "yes, proceed".

---

## 🧪 Testing Rules

- Every new public function: at least one test
- Accuracy: use `f.true_integral` + 3-sigma tolerance
- Reproducibility: same seed → `np.isclose(r1.mean, r2.mean)`
- Flow consistency (mandatory): `log q(x)` via forward pass == via inverse pass, atol=1e-6
- Never change a test assertion to make it pass — fix the code

---

## 🚀 Git Conventions

```
<type>(<scope>): short description

[body: physics motivation or implementation notes]
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `results` · `chore`
Scopes: `level1` · `level2` · `level3` · `flow` · `vegas` · `viz` · `ci` · `function`

- `main`: always green, `pytest tests/` passes
- Feature branches: `feat/level-{N}`
- Never force-push to `main`

---

## 📦 Dependencies

| Group | Packages | How to install |
|---|---|---|
| Core | `torch`, `numpy`, `scipy`, `matplotlib`, `pandas`, `vegas`, `tqdm` | bootstrap prompt |
| HEP | `scikit-hep`, `uproot` | `pip install -e ".[hep]"` |
| Dev | `pytest`, `pytest-cov`, `ruff`, `nbconvert`, `jupyter`, `ipykernel` | bootstrap prompt |

**Installation order matters.** Always `pip install torch --index-url ...` before `pip install -e .`.

---

*Last updated: March 2026 · nis-mc v0.1.0 · Fish shell aware*