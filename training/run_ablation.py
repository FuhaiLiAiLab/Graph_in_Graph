"""Fixed-parameter training entry point."""
from run import main
if __name__ == "__main__":
    main(['phenotype_only', 'gene_only', 'rna_only', 'methylation_only'])
