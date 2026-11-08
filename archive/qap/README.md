# QAP archive

The quantization-aware pruning (QAP) criterion line is closed.  This
directory keeps its code and results out of the active tree.

- `code/minitests/`, `code/visualize_plots/`: whole-file QAP minitests
  (`criterion_compare` with the eqap/kqap/wqap candidates, `criterion_anatomy`,
  `prune_scale_invariance`) and their plots, as of commit 965f0a5.
- `code/removed_snippets.py`: QAP functions and branches removed from shared
  files (`qap_score`, the pipeline branch, the obr_diag mid-tread tie test).
- `results/` (not committed): the matching JSON results and figures.

Findings, in short: within a quantization group QAP equals magnitude (the
mid-rise score `w^2 - eps^2` is monotone in `|w|`), so it only moves sparsity
across groups; under OBR it is within 1% of SparseGPT at W2-W4 and helps only
at W1.  At W2 mid-tread 76% of scores are exactly zero and ties decide every
row.
