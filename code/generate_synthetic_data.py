"""
Synthetic Data Generator for DeepSample Framework Thesis
Generates realistic synthetic datasets for DNN/LLM testing experiments
"""

import numpy as np
import pandas as pd
from scipy import stats
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sentiment_imdb(n_samples=10000):
    """Generate synthetic IMDb sentiment analysis dataset"""
    print(f"Generating {n_samples} synthetic IMDb reviews...")
    
    data = []
    for i in range(n_samples):
        # True label (0: negative, 1: positive)
        true_label = random.choice([0, 1])
        
        # Model prediction (with some error rate)
        # Simulate 85% accuracy
        if random.random() < 0.85:
            predicted_label = true_label
        else:
            predicted_label = 1 - true_label
        
        # Generate confidence score (higher for correct predictions)
        if predicted_label == true_label:
            # Correct predictions: higher confidence (0.6-0.99)
            confidence = np.random.beta(8, 2)
        else:
            # Incorrect predictions: lower confidence (0.4-0.7)
            confidence = np.random.beta(2, 3) * 0.6 + 0.4
        
        # Generate prediction probabilities
        prob_positive = confidence if predicted_label == 1 else (1 - confidence)
        prob_negative = 1 - prob_positive
        
        # Calculate prediction entropy
        probs = [prob_negative, prob_positive]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        # Generate surprise adequacy metrics (DSA/LSA)
        # Higher values indicate more novel/unusual inputs
        if predicted_label == true_label:
            dsa = np.random.gamma(2, 0.3)  # Lower for correct
            lsa = np.random.gamma(2, 0.4)
        else:
            dsa = np.random.gamma(4, 0.5)  # Higher for incorrect
            lsa = np.random.gamma(4, 0.6)
        
        data.append({
            'sample_id': f'imdb_{i:05d}',
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'prob_negative': prob_negative,
            'prob_positive': prob_positive,
            'entropy': entropy,
            'dsa': dsa,
            'lsa': lsa,
            'is_correct': int(predicted_label == true_label)
        })
    
    df = pd.DataFrame(data)
    return df

def generate_sentiment_sst2(n_samples=5000):
    """Generate synthetic SST-2 sentiment dataset"""
    print(f"Generating {n_samples} synthetic SST-2 samples...")
    
    data = []
    for i in range(n_samples):
        true_label = random.choice([0, 1])
        
        # Simulate 88% accuracy for SST-2
        if random.random() < 0.88:
            predicted_label = true_label
        else:
            predicted_label = 1 - true_label
        
        # Generate confidence (slightly different distribution than IMDb)
        if predicted_label == true_label:
            confidence = np.random.beta(9, 2)
        else:
            confidence = np.random.beta(2, 4) * 0.5 + 0.45
        
        prob_positive = confidence if predicted_label == 1 else (1 - confidence)
        prob_negative = 1 - prob_positive
        
        probs = [prob_negative, prob_positive]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        # Surprise adequacy for shorter texts (different distribution)
        if predicted_label == true_label:
            dsa = np.random.gamma(1.5, 0.25)
            lsa = np.random.gamma(1.5, 0.35)
        else:
            dsa = np.random.gamma(3.5, 0.45)
            lsa = np.random.gamma(3.5, 0.55)
        
        data.append({
            'sample_id': f'sst2_{i:05d}',
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'prob_negative': prob_negative,
            'prob_positive': prob_positive,
            'entropy': entropy,
            'dsa': dsa,
            'lsa': lsa,
            'is_correct': int(predicted_label == true_label)
        })
    
    df = pd.DataFrame(data)
    return df

def generate_vision_cifar(n_samples=8000):
    """Generate synthetic CIFAR-10 vision classification dataset"""
    print(f"Generating {n_samples} synthetic CIFAR-10 samples...")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    data = []
    for i in range(n_samples):
        true_label = random.randint(0, 9)
        
        # Simulate 82% accuracy for vision model
        if random.random() < 0.82:
            predicted_label = true_label
        else:
            # Incorrect prediction to a different class
            predicted_label = random.choice([c for c in range(10) if c != true_label])
        
        # Generate 10-class probability distribution
        probs = np.random.dirichlet(np.ones(10) * 0.5)
        
        # Boost the predicted class probability
        if predicted_label == true_label:
            probs[predicted_label] = np.random.beta(10, 2)
        else:
            probs[predicted_label] = np.random.beta(5, 3)
        
        # Renormalize
        probs = probs / probs.sum()
        
        confidence = probs[predicted_label]
        
        # Calculate entropy for 10 classes
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        # Surprise adequacy for vision
        if predicted_label == true_label:
            dsa = np.random.gamma(2.5, 0.4)
            lsa = np.random.gamma(2.5, 0.5)
        else:
            dsa = np.random.gamma(5, 0.7)
            lsa = np.random.gamma(5, 0.8)
        
        data.append({
            'sample_id': f'cifar_{i:05d}',
            'true_label': true_label,
            'true_class': class_names[true_label],
            'predicted_label': predicted_label,
            'predicted_class': class_names[predicted_label],
            'confidence': confidence,
            'entropy': entropy,
            'dsa': dsa,
            'lsa': lsa,
            'is_correct': int(predicted_label == true_label),
            **{f'prob_class_{j}': probs[j] for j in range(10)}
        })
    
    df = pd.DataFrame(data)
    return df

def generate_experimental_results():
    """Generate synthetic experimental results for all sampling methods"""
    print("Generating experimental results comparison...")
    
    methods = ['SRS', 'SUPS', 'RHC-S', 'SSRS', 'GBS', '2-UPS', 'DeepEST']
    datasets = ['IMDb', 'SST-2', 'CIFAR-10']
    sample_sizes = [50, 100, 200, 400, 800]
    aux_variables = ['Confidence', 'Entropy']
    
    data = []
    
    for dataset in datasets:
        # True accuracy for each dataset
        true_acc = {'IMDb': 0.85, 'SST-2': 0.88, 'CIFAR-10': 0.82}[dataset]
        
        for method in methods:
            for sample_size in sample_sizes:
                for aux_var in aux_variables:
                    # Simulate RMSE (lower for larger samples)
                    base_rmse = {
                        'SRS': 0.03, 'SUPS': 0.025, 'RHC-S': 0.028,
                        'SSRS': 0.04, 'GBS': 0.027, '2-UPS': 0.032, 'DeepEST': 0.029
                    }[method]
                    
                    # Adjust for sample size
                    rmse = base_rmse * (100 / sample_size) ** 0.5
                    rmse += np.random.normal(0, 0.002)
                    
                    # Estimated accuracy
                    estimated_acc = true_acc + np.random.normal(0, rmse)
                    
                    # Mispredictions found (higher for biased methods)
                    base_failures = {
                        'SRS': 20, 'SUPS': 85, 'RHC-S': 55,
                        'SSRS': 79, 'GBS': 45, '2-UPS': 50, 'DeepEST': 75
                    }[method]
                    
                    # Scale with sample size
                    failures_found = int(base_failures * (sample_size / 200) + np.random.normal(0, 5))
                    failures_found = max(0, failures_found)
                    
                    # Confidence interval width
                    ci_width = 1.96 * 2 * rmse
                    
                    data.append({
                        'method': method,
                        'dataset': dataset,
                        'sample_size': sample_size,
                        'auxiliary_variable': aux_var,
                        'true_accuracy': true_acc,
                        'estimated_accuracy': estimated_acc,
                        'rmse': rmse,
                        'failures_found': failures_found,
                        'ci_width': ci_width,
                        'ci_lower': estimated_acc - ci_width/2,
                        'ci_upper': estimated_acc + ci_width/2
                    })
    
    df = pd.DataFrame(data)
    return df

def main():
    """Generate all synthetic datasets"""
    print("=" * 60)
    print("DeepSample Framework - Synthetic Data Generation")
    print("=" * 60)
    
    # Generate datasets
    imdb_df = generate_sentiment_imdb(10000)
    sst2_df = generate_sentiment_sst2(5000)
    cifar_df = generate_vision_cifar(8000)
    results_df = generate_experimental_results()
    
    # Save to CSV
    print("\nSaving datasets...")
    imdb_df.to_csv('datasets/synthetic_sentiment_imdb.csv', index=False)
    print(f"✓ Saved: datasets/synthetic_sentiment_imdb.csv ({len(imdb_df)} samples)")
    
    sst2_df.to_csv('datasets/synthetic_sentiment_sst2.csv', index=False)
    print(f"✓ Saved: datasets/synthetic_sentiment_sst2.csv ({len(sst2_df)} samples)")
    
    cifar_df.to_csv('datasets/synthetic_vision_cifar.csv', index=False)
    print(f"✓ Saved: datasets/synthetic_vision_cifar.csv ({len(cifar_df)} samples)")
    
    results_df.to_csv('datasets/sampling_results_comparison.csv', index=False)
    print(f"✓ Saved: datasets/sampling_results_comparison.csv ({len(results_df)} results)")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary Statistics")
    print("=" * 60)
    
    print("\nIMDb Dataset:")
    print(f"  Total samples: {len(imdb_df)}")
    print(f"  Accuracy: {imdb_df['is_correct'].mean():.2%}")
    print(f"  Avg confidence: {imdb_df['confidence'].mean():.3f}")
    print(f"  Avg entropy: {imdb_df['entropy'].mean():.3f}")
    
    print("\nSST-2 Dataset:")
    print(f"  Total samples: {len(sst2_df)}")
    print(f"  Accuracy: {sst2_df['is_correct'].mean():.2%}")
    print(f"  Avg confidence: {sst2_df['confidence'].mean():.3f}")
    print(f"  Avg entropy: {sst2_df['entropy'].mean():.3f}")
    
    print("\nCIFAR-10 Dataset:")
    print(f"  Total samples: {len(cifar_df)}")
    print(f"  Accuracy: {cifar_df['is_correct'].mean():.2%}")
    print(f"  Avg confidence: {cifar_df['confidence'].mean():.3f}")
    print(f"  Avg entropy: {cifar_df['entropy'].mean():.3f}")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
