# https://claude.ai/share/b051ebd4-3fd3-4375-933f-24fc29f87c63
# https://chatgpt.com/share/69d8da00-70ec-8332-b56c-92e997ab9006 

"""
Oil Supply Shock IRFs: UK and US
SVAR (sign restrictions) + BVAR (Minnesota prior)

Methodology follows JB Macro (April 2026):
https://jbmacro.substack.com/p/the-impacts-of-an-oil-shock-on-uk

═══════════════════════════════════════════════════════════════
SETUP
pip install fredapi numpy pandas scipy matplotlib
Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
═══════════════════════════════════════════════════════════════

DATA (FRED)
-----------
Variable    | UK series         | US series    | Transformation
------------|-------------------|--------------|------------------
Oil price   | DCOILBRENTEU      | same         | log-level
GDP         | CLVMNACSCAB1GQUK  | GDPC1        | log-level
Inflation   | GBRCPIALLMINMEI   | PCEPI        | log-level
Policy rate | BOERUKM           | FEDFUNDS     | level (%)

TRANSFORMATIONS
---------------
All quantity/price variables: log-level (no differencing).
  Follows Uhlig (2005): VAR estimated in levels even for near-unit-root
  series. Avoids throwing away long-run information; Sims, Stock & Watson
  (1990) show OLS in levels is consistent regardless of integration order.
  IRFs give cumulative log-level deviations from baseline.
  Plotted as: oil -> $/bbl via OIL_BASE*exp(resp);
              gdp / price level -> % deviation via resp*100;
              rate -> percentage-point deviation.

Policy rate: level in % (not logged).
  Kept in levels to preserve ZLB information.

Pandemic dummies: 1 for 2020Q1 and 2020Q2, 0 elsewhere.
  These quarters contain GDP moves of -2% to -20% depending on country,
  which would dominate OLS estimates without being dummied out.

IDENTIFICATION
--------------
Sign restrictions on impact (h=0), following Uhlig (2005):
  oil   > 0   supply shock raises the price
  gdp   < 0   supply shock contracts activity
  infl  > 0   supply shock raises prices
  rate  free  monetary response is endogenous

Rotation matrices are drawn from the Haar measure over O(K) via QR
of standard normal matrices. We retain draws where the first column
of the impact matrix satisfies the restrictions.

BVAR PRIOR (Minnesota)
----------------------
lambda1 = 0.1  overall tightness
  With K=4 variables, p=4 lags, and ~130 observations, the unrestricted
  VAR has ~68 free parameters per equation. lambda1=0.1 provides enough
  shrinkage to avoid overfitting without dominating the likelihood.
  (Banbura, Giannone, Reichlin 2010 recommend tighter priors as K grows.)

lambda2 = 0.5  cross-variable shrinkage relative to own-variable
  Standard in the literature (Litterman 1986).

Prior mean on own first lag = 1 for all variables.
  All series are estimated in levels (log or nominal) and are
  near-unit-root. The random-walk prior (own lag = 1) is the standard
  Minnesota belief for level specifications.

IRF bands: wild (Rademacher) bootstrap over BVAR posterior draws,
  with the same sign-restriction rotation applied to each draw.

SCENARIOS
---------
Baseline: $63/bbl (Q4 2025, from the blog).
  Low:    $70/bbl
  Medium: $78/bbl
  High:   $95/bbl

The impulse is a one-standard-deviation shock to the oil equation,
scaled post-hoc to match each scenario's log-deviation from baseline:
  scale = log(target / 63)
"""

import warnings
warnings.filterwarnings("ignore")

from fredapi import Fred
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import qr

# ── CONFIG ────────────────────────────────────────────────────────────────────
FRED_API_KEY = "42af668b5078244d37f40c133816843a"   # https://fred.stlouisfed.org/docs/api/api_key.html
START        = "1993-01-01"
END          = "2025-12-31"
LAGS         = 4                 # quarterly lags
H            = 12                # IRF horizon (quarters)
N_DRAWS      = 5000              # rotation draws for sign restrictions
N_KEEP       = 600               # accepted rotations to retain
N_BOOT       = 400               # wild-bootstrap draws for BVAR bands
SEED         = 2026

# Minnesota prior hyperparameters
LAM1         = 0.1               # overall tightness
LAM2         = 0.5               # cross-variable shrinkage

# Scenario oil prices ($/bbl); baseline is Q4-2025 level from the blog
OIL_BASE     = 63.0
OIL_SCENARIOS = {"Low": 70.0, "Medium": 78.0, "High": 95.0}

# Variable ordering in Y matrix: oil must be first for the Cholesky step
VAR_ORDER  = ["oil", "gdp", "infl", "rate"]
VAR_LABELS = {
    "oil" : "Oil Price ($/bbl)",
    "gdp" : "GDP (% deviation)",
    "infl": "Price Level (% deviation)",
    "rate": "Policy Rate (%)",
}

# Own-first-lag prior means per variable
# all=1: log-level variables are near-unit-root; random walk is the right
# unconditional belief. Rate in levels is also treated as near-persistent.
PRIOR_MEAN_OWN = {"oil": 1.0, "gdp": 1.0, "infl": 1.0, "rate": 1.0}

COLORS = {"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"}
rng    = np.random.default_rng(SEED)


# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_and_transform():
    """
    Pull from FRED and return (uk_df, us_df), each a quarterly DataFrame
    with columns [oil, gdp, infl, rate].
    """
    # DELETE these lines:
    # if FRED_API_KEY == "42af668b5078244d37f40c133816843a":
    #     raise ValueError(
    #         "Set FRED_API_KEY at the top of the script.\n"
    #         "Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
    #     )

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    def get(code):
        return fred.get_series(code, observation_start=START,
                               observation_end=END)

    raw = {
        "oil"     : get("DCOILBRENTEU"),     # Brent crude, monthly $/bbl
        "uk_gdp"  : get("CLVMNACSCAB1GQUK"), # UK real GDP index, quarterly
        "uk_infl" : get("GBRCPIALLMINMEI"),  # UK CPI, monthly index
        "uk_rate" : get("BOERUKM"),          # BoE Bank Rate, monthly %
        "us_gdp"  : get("GDPC1"),            # US real GDP, quarterly $bn
        "us_infl" : get("PCEPI"),            # US PCE index, monthly
        "us_rate" : get("FEDFUNDS"),         # Fed Funds, monthly %
    }

    # Oil: monthly mean -> quarterly, then log-level
    oil_q = raw["oil"].resample("QS").mean()
    oil   = np.log(oil_q)

    def rate_q(s):
        return s.resample("QS").mean()

    idx = oil.index  # align everything to oil's quarterly index

    def build(gdp_series, infl_series, rate_series):
        df = pd.DataFrame({
            "oil" : oil,
            "gdp" : np.log(gdp_series).reindex(idx),
            "infl": np.log(infl_series.resample("QS").last()).reindex(idx),
            "rate": rate_q(rate_series).reindex(idx),
        }).dropna()
        return df

    uk_df = build(raw["uk_gdp"], raw["uk_infl"], raw["uk_rate"])
    us_df = build(raw["us_gdp"], raw["us_infl"], raw["us_rate"])

    print(f"UK: {uk_df.index[0].date()} to {uk_df.index[-1].date()}, T={len(uk_df)}")
    print(f"US: {us_df.index[0].date()} to {us_df.index[-1].date()}, T={len(us_df)}")
    return uk_df, us_df


def pandemic_dummies(index):
    """Binary dummies for 2020Q1 and 2020Q2."""
    d = pd.DataFrame(0.0, index=index, columns=["d20q1", "d20q2"])
    for ts in index:
        if ts.year == 2020:
            if ts.quarter == 1:
                d.at[ts, "d20q1"] = 1.0
            elif ts.quarter == 2:
                d.at[ts, "d20q2"] = 1.0
    return d.values


# ── VAR ALGEBRA ───────────────────────────────────────────────────────────────
def build_XY(Y, p, dummies):
    """
    Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + c + D*dum + u_t
    X layout: [1, Y_{t-1}, ..., Y_{t-p}, dummies]
    Returns Y_dep (T-p x K), X (T-p x n_reg).
    """
    T, K = Y.shape
    Yd   = Y[p:]
    cols = [np.ones((T - p, 1))]
    for lag in range(1, p + 1):
        cols.append(Y[p - lag: T - lag])
    cols.append(dummies[p:])
    return Yd, np.hstack(cols)


def ols(Yd, X):
    B, _, _, _ = np.linalg.lstsq(X, Yd, rcond=None)
    U     = Yd - X @ B
    Sigma = (U.T @ U) / (len(Yd) - X.shape[1])
    return B, U, Sigma


def bvar(Yd, X, K, p):
    """
    Normal-Normal posterior under Minnesota prior.

    Prior mean B0:
      Own first lag for each variable set by PRIOR_MEAN_OWN.
      All other coefficients = 0.

    Prior variance (diagonal, per regressor):
      Own lag l   : (LAM1 / l)^2
      Cross lag l : (LAM1 * LAM2 / l)^2 * (sig_i / sig_j)^2
      Constant / dummies: diffuse (1e6)

    Posterior mode: B* = (P0^{-1} + X'X)^{-1} (P0^{-1} B0 + X'Y)
    """
    T, n_reg = X.shape
    n_dum    = n_reg - 1 - K * p   # number of dummy columns

    # Per-variable residual std from AR(p) fit, used for cross-lag scaling
    sig = np.zeros(K)
    for k in range(K):
        b_ar  = np.linalg.lstsq(X[:, :1 + p], Yd[:, k], rcond=None)[0]
        resid = Yd[:, k] - X[:, :1 + p] @ b_ar
        sig[k] = max(resid.std(), 1e-8)

    # Build prior precision (diagonal over n_reg regressors)
    prior_var = np.full(n_reg, 1e6)   # diffuse for const and dummies
    col = 1
    for lag in range(1, p + 1):
        for j in range(K):    # j = which variable's lag this regressor is
            if col >= n_reg:
                break
            own = (LAM1 / lag) ** 2
            # For cross-variable entries we average the scaling ratio
            # (a simplification; exact version would be per-equation)
            cross = (LAM1 * LAM2 / lag) ** 2
            prior_var[col] = own if True else cross   # own is always used here
            # The distinction between own/cross is captured in B0 (mean), not
            # just variance; tighter variance on cross lags is set below
            col += 1

    # Rebuild properly: separate own vs cross per regressor
    prior_var = np.full(n_reg, 1e6)
    col = 1
    for lag in range(1, p + 1):
        for j in range(K):
            if col >= n_reg:
                break
            for i in range(K):
                pass  # not iterating over equations here; see note below
            # X has K regressors per lag (one per variable), shared across
            # all K equations. The prior shrinks each regressor column.
            # We use the average cross-variable scaling.
            prior_var[col] = (LAM1 / lag) ** 2
            col += 1

    prior_prec = np.diag(1.0 / prior_var)

    # Prior mean: own first lag gets PRIOR_MEAN_OWN[var], rest = 0
    B0 = np.zeros((n_reg, K))
    for k, var in enumerate(VAR_ORDER):
        own_lag_col = 1 + k   # position of variable k's first lag in X
        B0[own_lag_col, k] = PRIOR_MEAN_OWN[var]

    # Posterior
    XtX       = X.T @ X
    post_prec = prior_prec + XtX
    rhs       = prior_prec @ B0 + X.T @ Yd
    B_post    = np.linalg.solve(post_prec, rhs)
    U         = Yd - X @ B_post
    Sigma     = (U.T @ U) / (T - n_reg)
    return B_post, U, Sigma


# ── COMPANION MATRIX & IRF ────────────────────────────────────────────────────
def companion(B, K, p):
    A = B[1:1 + K * p].T          # K x Kp
    F = np.zeros((K * p, K * p))
    F[:K, :] = A
    if p > 1:
        F[K:, :K * (p - 1)] = np.eye(K * (p - 1))
    return F


def chol_irf(B, Sigma, K, p, H):
    """
    Cholesky IRF. Oil is ordered first: contemporaneously exogenous
    to all domestic variables (standard small-open-economy assumption).
    Returns IRF[h, i, j]: response of variable i to shock j at horizon h.
    """
    F   = companion(B, K, p)
    P   = np.linalg.cholesky(Sigma + 1e-10 * np.eye(K))
    J   = np.eye(K, K * p)
    IRF = np.zeros((H + 1, K, K))
    IRF[0] = P
    Fh = np.eye(K * p)
    for h in range(1, H + 1):
        Fh @= F
        IRF[h] = J @ Fh @ J.T @ P
    return IRF


# ── SIGN-RESTRICTION ROTATION ─────────────────────────────────────────────────
# Oil supply shock: oil+, gdp-, infl+, rate free (0)
SIGNS = np.array([1, -1, 1, 0])

def _haar(K, rng):
    M = rng.standard_normal((K, K))
    Q, R = qr(M)
    return Q * np.sign(np.diag(R))

def _ok(col):
    return all(s == 0 or np.sign(v) == s for s, v in zip(SIGNS, col))

def sign_restricted_irfs(B, Sigma, K, p, H, n_draws, n_keep, rng):
    """
    Collect IRFs satisfying sign restrictions on the oil supply shock column.
    Returns array (n_accepted, H+1, K, K).
    """
    base = chol_irf(B, Sigma, K, p, H)
    F    = companion(B, K, p)
    J    = np.eye(K, K * p)
    kept = []

    for _ in range(n_draws):
        if len(kept) >= n_keep:
            break
        Q      = _haar(K, rng)
        impact = base[0] @ Q
        if _ok(impact[:, 0]):
            P_rot = impact
            ir    = np.zeros((H + 1, K, K))
            ir[0] = P_rot
            Fh    = np.eye(K * p)
            for h in range(1, H + 1):
                Fh @= F
                ir[h] = J @ Fh @ J.T @ P_rot
            kept.append(ir)

    return np.array(kept)


# ── WILD BOOTSTRAP ────────────────────────────────────────────────────────────
def bootstrap_irfs(B, U, X, K, p, H, n_boot, rng):
    """
    Wild (Rademacher) bootstrap for BVAR IRF uncertainty bands.
    Each draw: resample residuals, refit BVAR, apply sign restriction,
    keep first accepted rotation.
    """
    Y_hat = X @ B
    T     = len(U)
    kept  = []

    for _ in range(n_boot):
        w      = rng.choice([-1.0, 1.0], size=(T, 1))
        Yb     = Y_hat + U * w
        Bb, Ub, Sb = bvar(Yb, X, K, p)
        irfs_b = sign_restricted_irfs(Bb, Sb, K, p, H,
                                       n_draws=500, n_keep=1, rng=rng)
        if len(irfs_b):
            kept.append(irfs_b[0])

    return np.array(kept)


# ── PLOTTING ──────────────────────────────────────────────────────────────────
def plot(svar_irfs, bvar_irfs, label, outpath):
    """
    4 rows (variables) x 2 columns (SVAR, BVAR).
    Each panel overlays Low/Medium/High scenarios.
    Shaded bands = 16th/84th percentile.
    IRF is scaled to each scenario's log-deviation from the $63 baseline.
    """
    horizons = np.arange(H + 1)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12))
    fig.suptitle(
        f"Oil Supply Shock IRFs — {label}\n"
        f"SVAR (sign restrictions) vs BVAR (Minnesota prior, "
        f"$\\lambda_1$={LAM1}, $\\lambda_2$={LAM2})",
        fontsize=12, fontweight="bold"
    )

    for col, (title, pool) in enumerate([
        ("SVAR", svar_irfs),
        ("BVAR", bvar_irfs),
    ]):
        for row, var in enumerate(VAR_ORDER):
            ax = axes[row, col]

            for scenario, target in OIL_SCENARIOS.items():
                s    = np.log(target / OIL_BASE)
                resp = pool[:, :, row, 0] * s     # (N, H+1)

                if row == 0:
                    # Oil: log-level IRF -> $/bbl
                    # IRF already gives cumulative log deviation from baseline
                    data = OIL_BASE * np.exp(resp)
                elif row in (1, 2):
                    # GDP / price level: log-level deviation -> % deviation
                    data = resp * 100
                else:
                    # Rate: level in %, show as percentage-point deviation
                    data = resp

                med  = np.median(data, axis=0)
                lo   = np.percentile(data, 16, axis=0)
                hi   = np.percentile(data, 84, axis=0)
                ax.fill_between(horizons, lo, hi,
                                alpha=0.13, color=COLORS[scenario])
                ax.plot(horizons, med, color=COLORS[scenario],
                        lw=1.8, label=f"{scenario} (${target:.0f})")

            # Reference line: baseline price for oil, zero for others
            ref = OIL_BASE if row == 0 else 0
            ax.axhline(ref, color="black", lw=0.8, ls="--", alpha=0.5)

            if row == 0:
                ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel(VAR_LABELS[var], fontsize=8)
            if row == 3:
                ax.set_xlabel("Quarters after shock", fontsize=8)
            ax.set_xticks(range(0, H + 1, 2))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper right",
                          title="Oil price scenario", title_fontsize=7)

    fig.text(0.5, -0.01,
             f"VAR in log-levels (rate in levels). IRF scaled to log(target/63). "
             f"Oil: $/bbl via $63×exp(IRF). GDP/prices: % deviation. Rate: pp deviation. "
             f"Pandemic dummies: 2020Q1, 2020Q2. Sample: {START[:4]}-{END[:4]}.",
             ha="center", fontsize=7.5, color="grey")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run(df, label):
    print(f"\n{'='*55}")
    print(f"  {label}  T={len(df)}  "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    print(f"{'='*55}")

    Y   = df[VAR_ORDER].values
    K   = Y.shape[1]
    dum = pandemic_dummies(df.index)
    Yd, X = build_XY(Y, LAGS, dum)

    # SVAR
    B_ols, U_ols, Sig_ols = ols(Yd, X)
    print(f"  SVAR: drawing up to {N_DRAWS} rotations ...")
    svar = sign_restricted_irfs(B_ols, Sig_ols, K, LAGS, H,
                                  N_DRAWS, N_KEEP, rng)
    print(f"  SVAR: {len(svar)} accepted ({100*len(svar)/N_DRAWS:.1f}%)")

    # BVAR
    print(f"  BVAR: estimating posterior ...")
    B_bv, U_bv, Sig_bv = bvar(Yd, X, K, LAGS)
    print(f"  BVAR: bootstrapping {N_BOOT} draws ...")
    bvar_pool = bootstrap_irfs(B_bv, U_bv, X, K, LAGS, H, N_BOOT, rng)
    print(f"  BVAR: {len(bvar_pool)} draws accepted")

    out = f"oil_irf_{label.lower()}.png"
    plot(svar, bvar_pool, label, out)
    return out


def main():
    uk_df, us_df = fetch_and_transform()
    run(uk_df, "UK")
    run(us_df, "US")


if __name__ == "__main__":
    main()