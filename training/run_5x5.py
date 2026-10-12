"""Fixed-parameter training entry point."""
from run import main
if __name__ == "__main__":
    main(['full', 'phenotype_only', 'gene_only', 'rna_only', 'methylation_only', 'gat', 'gin'])
