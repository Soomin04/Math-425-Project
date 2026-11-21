"""Fit Poisson and negative-binomial PMFs to PBMC3k raw gene counts.

Run after prepare_pbmc3k.py:
    python plot_LTB_GAPDH_pmfs.py
Dependencies: numpy pandas scipy matplotlib
Outputs go to pbmc3k_results/model_fits/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import poisson, nbinom

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'cd4_t_cells_similar_depth_raw_counts.csv'
if not INPUT.exists():
    INPUT = BASE / 'pbmc3k_results' / 'cd4_t_cells_similar_depth_raw_counts.csv'
OUT = BASE / 'model_fits_two_genes'
OUT.mkdir(parents=True, exist_ok=True)
GENES = ['LTB', 'GAPDH']

if not INPUT.exists():
    raise FileNotFoundError(f'{INPUT} not found. Run prepare_pbmc3k.py first.')
df = pd.read_csv(INPUT)
if len(df) < 2:
    raise ValueError('Need at least two cells for fitting.')
rows = []
for gene in GENES:
    x = df[gene].to_numpy(dtype=int)
    if np.any(x < 0) or np.any(x != df[gene].to_numpy()):
        raise ValueError(f'Expected nonnegative integer counts for {gene}')
    n = len(x)
    mu = float(x.mean())
    if mu == 0:
        print(f'Skipping {gene}: all counts zero')
        continue
    variance = float(x.var(ddof=1))

    # Poisson MLE: lambda_hat = mean(x)
    loglik_poisson = float(poisson.logpmf(x, mu).sum())

    # Negative-binomial MLE, parameterized by mean mu and size r:
    # var = mu + mu^2 / r; p = r / (r + mu).
    # For fixed r, MLE of mu is the sample mean. Optimize r on log scale.
    def negative_loglik(log_r):
        r = float(np.exp(log_r))
        p = r / (r + mu)
        return -float(nbinom.logpmf(x, r, p).sum())

    optimum = minimize_scalar(negative_loglik, bounds=(-10, 16), method='bounded', options={'xatol': 1e-8})
    if not optimum.success:
        raise RuntimeError(f'NB optimization failed for {gene}')
    rhat = float(np.exp(optimum.x))
    phat = rhat / (rhat + mu)
    loglik_nb = -float(optimum.fun)
    # If the NB optimum is indistinguishable from the Poisson limit, report it as such.
    at_poisson_limit = (optimum.x > 15.9 or loglik_nb <= loglik_poisson + 1e-7)
    if at_poisson_limit:
        loglik_nb = loglik_poisson

    k = np.arange(int(x.max()) + 1)
    empirical = np.bincount(x, minlength=len(k)) / n
    pois_pmf = poisson.pmf(k, mu)
    nb_pmf = nbinom.pmf(k, rhat, phat)
    pmf_df = pd.DataFrame({'count_k': k, 'empirical_pmf': empirical,
                           'poisson_pmf': pois_pmf, 'negative_binomial_pmf': nb_pmf})
    pmf_df.to_csv(OUT / f'{gene}_pmfs.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(k, empirical, width=0.75, alpha=0.48, label='Observed empirical PMF')
    ax.plot(k, pois_pmf, marker='o', linewidth=1.7, label=f'Poisson (mean={mu:.2f})')
    ax.plot(k, nb_pmf, marker='s', linewidth=1.7, label=f'Negative binomial (r={rhat:.2f})')
    ax.set_xlabel('Raw gene UMI count per CD4 T cell')
    ax.set_ylabel('Probability')
    ax.set_title(f'{gene}: observed vs fitted PMFs (similar-depth CD4 T cells)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f'{gene}_pmf_comparison.png', dpi=200)
    plt.close(fig)

    rows.append({'gene': gene, 'cells': n, 'sample_mean': mu, 'sample_variance': variance,
                 'dispersion_index': variance / mu, 'observed_zero_fraction': float((x==0).mean()),
                 'poisson_zero_probability': float(poisson.pmf(0, mu)),
                 'nb_zero_probability': float(nbinom.pmf(0, rhat, phat)),
                 'nb_r_mle': rhat, 'nb_p_mle': phat,
                 'poisson_loglik': loglik_poisson, 'nb_loglik': loglik_nb,
                 'poisson_AIC': 2 - 2*loglik_poisson, 'nb_AIC': 4 - 2*loglik_nb,
                 'nb_poisson_limit': at_poisson_limit})

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'model_fit_summary.csv', index=False)
print(summary.round(4).to_string(index=False))
print(f'\nSaved PMF data, plots, and model summary in {OUT}')
print('Note: both models ignore residual depth differences within the filtered cell subset.')
