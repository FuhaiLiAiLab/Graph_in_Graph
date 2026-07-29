import os
import json
import random
import time
import numpy as np

# Quick HPO test based on Optuna findings
def quick_hpo_test():
    print("Quick HPO Test based on Optuna findings")
    
    # Load Optuna results
    optuna_file = "optuna_try2_phase3_results.json"
    if os.path.exists(optuna_file):
        with open(optuna_file, 'r') as f:
            optuna_results = json.load(f)
        
        best_result = optuna_results[0]  # Top ranked
        best_params = best_result['params']
        best_mean_acc = best_result['mean_acc']
        
        print(f"Optuna best mean accuracy: {best_mean_acc:.4f}")
        print(f"Optuna best params: {best_params}")
        
        # Extract key parameters
        optuna_lr = best_params['lr']
        optuna_weight_decay = best_params['weight_decay']
        optuna_gene_dim = best_params['gene_dim']
        optuna_gig_dim = best_params['gig_dim']
        
        print(f"\nKey parameters from Optuna:")
        print(f"  lr: {optuna_lr}")
        print(f"  weight_decay: {optuna_weight_decay}")
        print(f"  gene_dim: {optuna_gene_dim}")
        print(f"  gig_dim: {optuna_gig_dim}")
    
    # Define search space around Optuna best
    search_space = {
        'lr': [optuna_lr * 0.1, optuna_lr, optuna_lr * 10],
        'weight_decay': [optuna_weight_decay * 0.1, optuna_weight_decay, optuna_weight_decay * 10],
        'gene_hidden_dim': [optuna_gene_dim, 12, 18],
        'gig_hidden_dim': [optuna_gig_dim, 24, 32],
        'class_weight': [0.7, 0.8, 0.9, 0.95],
        'ortho_weight': [0.001, 0.005, 0.01, 0.05],
        'link_weight': [0.001, 0.005, 0.01, 0.05],
        'dropout': [0.01, 0.05, 0.1]
    }
    
    print(f"\nSearch space around Optuna best:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    
    # We'll test 5 random combinations
    n_trials = 5
    print(f"\nTesting {n_trials} random combinations...")
    
    results = []
    
    for trial in range(1, n_trials + 1):
        params = {}
        for key, values in search_space.items():
            params[key] = random.choice(values)
        
        print(f"\nTrial {trial}/{n_trials}: {params}")
        
        # In a real run, we would call run_model here
        # For now, just simulate or run actual model
        # We'll create a placeholder result
        simulated_accuracy = random.uniform(0.1, 0.3)  # Placeholder
        
        result = {
            'params': params,
            'accuracy': simulated_accuracy,
            'trial': trial
        }
        results.append(result)
        
        print(f"  Simulated accuracy: {simulated_accuracy:.4f}")
        
        # If we get > 0.8 in simulation, note it
        if simulated_accuracy > 0.8:
            print(f"  WOULD EXCEED TARGET! (simulated)")
    
    # Find best
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest simulated accuracy: {best_result['accuracy']:.4f}")
        print(f"Best parameters: {best_result['params']}")
    
    return results

if __name__ == "__main__":
    quick_hpo_test()