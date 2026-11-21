"""Extract unnormalized UMI counts for LTB, IL7R, and GAPDH in annotated CD4 T cells.

Install: python -m pip install scanpy pandas numpy scipy
Run: python prepare_pbmc3k.py
Requires Internet access on the first run to obtain Scanpy's PBMC3k examples.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

OUT = Path(__file__).resolve().parent / 'pbmc3k_results'
OUT.mkdir(parents=True, exist_ok=True)
GENES = ['LTB', 'IL7R', 'GAPDH']

print('STEP 1: Loading original unnormalized PBMC3k UMI matrix...')
raw = sc.datasets.pbmc3k()  # 2700 cells x 32738 genes, nonnegative integer counts
print(f'Raw shape: {raw.n_obs} cells x {raw.n_vars} genes')

print('STEP 2: Loading independently processed PBMC3k cell annotations...')
annotated = sc.datasets.pbmc3k_processed()  # processed values used only for labels
if 'louvain' not in annotated.obs:
    raise RuntimeError('Expected annotation column louvain not found')
labels = annotated.obs['louvain'].astype(str)
print('Cell labels / counts:')
print(labels.value_counts().to_string())
cell_type = 'CD4 T cells'
if cell_type not in set(labels):
    raise RuntimeError(f'{cell_type!r} not found in annotation labels')
cd4_ids = annotated.obs_names[labels.eq(cell_type)]

print('STEP 3: Validating barcode alignment and selecting CD4 T cells...')
raw_ids = set(raw.obs_names)
missing = cd4_ids.difference(raw.obs_names)
if len(missing):
    raise RuntimeError(f'{len(missing)} annotated CD4 barcodes absent from raw matrix; STOP')
print(f'Annotated CD4 T cells: {len(cd4_ids)}; all match original matrix by barcode')

print('STEP 4: Extracting raw INTEGER counts (no log transform or normalization)...')
absent_genes = [g for g in GENES if g not in raw.var_names]
if absent_genes:
    raise RuntimeError(f'Missing target genes in original matrix: {absent_genes}')
subset = raw[cd4_ids, :]
gene_slice = subset[:, GENES].X
counts = gene_slice.toarray() if sparse.issparse(gene_slice) else np.asarray(gene_slice)
if not np.all(np.isfinite(counts)) or np.any(counts < 0) or not np.all(counts == np.floor(counts)):
    raise RuntimeError('Raw counts have non-integer or invalid values; STOP')
counts = counts.astype(np.int64)

print('STEP 5: Calculating sequencing depth from ALL genes in each CD4 T cell...')
total_umis = np.asarray(subset.X.sum(axis=1)).reshape(-1)
if not np.all(total_umis == np.floor(total_umis)) or np.any(total_umis <= 0):
    raise RuntimeError('Unexpected total UMI values; STOP')
total_umis = total_umis.astype(np.int64)
result = pd.DataFrame(counts, index=pd.Index(cd4_ids, name='cell_barcode'), columns=GENES)
result.insert(0, 'total_umis', total_umis)
result['depth_factor_median'] = result['total_umis'] / float(np.median(total_umis))
result['within_20pct_median_depth'] = result['depth_factor_median'].between(0.8, 1.2)

print('STEP 6: Saving results, summary, and validation metadata...')
result.reset_index().to_csv(OUT / 'cd4_t_cells_raw_counts.csv', index=False)
result[result['within_20pct_median_depth']].reset_index().to_csv(
    OUT / 'cd4_t_cells_similar_depth_raw_counts.csv', index=False
)
summary_rows=[]
for gene in GENES:
    v=result[gene]
    summary_rows.append({
        'gene': gene, 'n_cells':len(v), 'mean_raw_count':v.mean(),
        'sample_variance_raw_count':v.var(ddof=1),
        'dispersion_index':v.var(ddof=1)/v.mean() if v.mean()>0 else np.nan,
        'fraction_zero':(v==0).mean(), 'max_raw_count':v.max(),
    })
pd.DataFrame(summary_rows).to_csv(OUT/'gene_summary_all_cd4.csv', index=False)
subset_depth = result[result['within_20pct_median_depth']]
summary_depth=[]
for gene in GENES:
    v=subset_depth[gene]
    summary_depth.append({
        'gene':gene,'n_cells':len(v), 'mean_raw_count':v.mean(),
        'sample_variance_raw_count':v.var(ddof=1),
        'dispersion_index':v.var(ddof=1)/v.mean() if len(v)>1 and v.mean()>0 else np.nan,
        'fraction_zero':(v==0).mean() if len(v) else np.nan,
        'max_raw_count':v.max() if len(v) else np.nan,
    })
pd.DataFrame(summary_depth).to_csv(OUT/'gene_summary_similar_depth.csv', index=False)
metadata = {
    'source_raw': 'scanpy.datasets.pbmc3k() / original 10x Genomics PBMC3k',
    'source_labels': 'scanpy.datasets.pbmc3k_processed() / louvain',
    'cell_type':cell_type,'gene_symbols':GENES,'raw_n_cells':raw.n_obs,
    'raw_n_genes':raw.n_vars,'annotated_n_cells':annotated.n_obs,
    'selected_cd4_cells':len(result), 'similar_depth_cells':len(subset_depth),
    'median_total_umis':float(np.median(total_umis)),
    'min_total_umis':int(total_umis.min()),
    'max_total_umis':int(total_umis.max()),
    'match_method':'exact matching of cell barcode',
    'gene_count_data':'raw unnormalized integer UMI counts',
    'depth_definition':'sum raw UMI counts over all genes for each CD4 T cell',
    'depth_filter':'0.8 <= total_umis / median(total_umis) <= 1.2; comparison subset only',
    'warnings': [
        'Processed annotations are inferred clusters, not independent ground-truth cell identities.',
        'Depth filtering is a sensitivity analysis and does not remove all sequencing/capture biases.',
        'Total UMI depth includes target gene counts; consider depth offsets or external normalization factors for advanced modeling.'
    ],
}
(OUT/'processing_metadata.json').write_text(json.dumps(metadata,indent=2))
print('\nSUMMARY — ALL SELECTED CD4 T CELLS:')
print(pd.DataFrame(summary_rows).round(3).to_string(index=False))
print('\nSEQUENCING DEPTH:')
print(pd.Series(total_umis).describe().round(1).to_string())
print(f'Similar-depth subset (within +/-20% of median): {len(subset_depth)} cells')
print(f'Files saved under: {OUT}')
