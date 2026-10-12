# Current revision

- Replaced obsolete training entry points with current Full, modality ablations and independent GAT/GIN implementations and fixed parameter files.
- Preserved family-separated folds, train-only downstream preprocessing, graph policies, weighted/unweighted loss distinctions and validation checkpoint rules.
- Added portable complete seed/fold training and strict checkpoint recovery without historical experiment-directory dependencies.
- Updated RNA-weighted node scoring, edge export and both Shiny apps; preserve signed values and identity-aligned network averaging.
- Removed hyperparameter search code and results, predictive-score reporting tools, historical notebooks, regenerated figures and incompatible legacy model entry points. Git history retains the prior revision.
- Added synthetic regression tests and current environment/data instructions. Runtime data, predictions, checkpoints and audit logs are excluded from source control.
