# -*- coding: utf-8 -*-
"""
Plant Disease Classification - XGBoost and Random Forest Training
Trains XGBoost and Random Forest classifiers using the same 29 features as NODE.
Uses pre-extracted features from HDF5 files.

GPU ACCELERATION:
- Random Forest uses cuML's GPU-accelerated implementation for faster training
- XGBoost uses CPU (tree_method='hist')
"""

import os
import numpy as np
import h5py
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from cuml.ensemble import RandomForestClassifier  # GPU-accelerated Random Forest
import cupy as cp  # For GPU array handling
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: Load Features from HDF5
# ============================================================================

def load_features_hdf5(file_path):
    """
    Load features and labels from HDF5 file.
    
    Returns:
        features: numpy array of shape (n_samples, n_features)
        plant_labels: list of plant type labels
        disease_labels: list of disease labels
    """
    print(f"\nLoading features from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        plant_labels = [s for s in f['plant_labels'][:]]
        disease_labels = [s for s in f['disease_labels'][:]]
    
    print(f"✓ Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"  Plants: {len(set(plant_labels))} classes")
    print(f"  Diseases: {len(set(disease_labels))} classes")
    
    return features, plant_labels, disease_labels


# ============================================================================
# SECTION 2: Confusion Matrix Visualization
# ============================================================================

def plot_confusion_matrix(cm, class_names, title, output_path, normalize=False):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: confusion matrix array
        class_names: list of class names
        title: plot title
        output_path: path to save figure
        normalize: whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(max(10, len(class_names) * 0.8), max(8, len(class_names) * 0.6)))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


def save_confusion_matrix_csv(cm, class_names, output_path):
    """Save confusion matrix as CSV."""
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(output_path)
    print(f"✓ Confusion matrix CSV saved to {output_path}")


# ============================================================================
# SECTION 3: XGBoost Classifier Training
# ============================================================================

def train_xgboost_classifier(X_train, y_train, X_test, y_test, 
                            class_names, classifier_name, 
                            output_dir='.', config=None):
    """
    Train XGBoost classifier.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        class_names: list of class names
        classifier_name: 'plant' or 'disease'
        output_dir: directory to save outputs
        config: optional dict of hyperparameters (from Optuna)
    """
    print("\n" + "="*70)
    print(f"Training {classifier_name.upper()} Classifier - XGBoost")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {len(class_names)}")
    
    # Default config or use provided
    if config is None:
        config = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
    
    print(f"\nXGBoost config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Train model
    print("\nTraining XGBoost...")
    model = XGBClassifier(**config)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Predictions
    print("  Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ XGBoost Accuracy: {accuracy:.4f}")
    
    # Print metrics
    print("\n" + "─"*70)
    print("FINAL XGBoost METRICS")
    print("─"*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)
    
    # Save confusion matrices
    prefix = f"{classifier_name}_xgboost"
    
    # Raw counts
    plot_confusion_matrix(
        cm, class_names,
        f'{classifier_name.title()} Classification - Confusion Matrix (XGBoost)',
        os.path.join(output_dir, f'{prefix}_confusion_matrix_raw.png'),
        normalize=False
    )
    
    # Normalized
    plot_confusion_matrix(
        cm, class_names,
        f'{classifier_name.title()} Classification - Confusion Matrix Normalized (XGBoost)',
        os.path.join(output_dir, f'{prefix}_confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Save CSV
    save_confusion_matrix_csv(
        cm, class_names,
        os.path.join(output_dir, f'{prefix}_confusion_matrix.csv')
    )
    
    print("─"*70)
    
    # Return results
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


# ============================================================================
# SECTION 4: Random Forest Classifier Training
# ============================================================================

def train_random_forest_classifier(X_train, y_train, X_test, y_test, 
                                  class_names, classifier_name, 
                                  output_dir='.', config=None):
    """
    Train Random Forest classifier using cuML GPU-accelerated implementation.
    
    Args:
        X_train, y_train: training data
        X_test, y_test: test data
        class_names: list of class names
        classifier_name: 'plant' or 'disease'
        output_dir: directory to save outputs
        config: optional dict of hyperparameters (from Optuna)
    """
    print("\n" + "="*70)
    print(f"Training {classifier_name.upper()} Classifier - Random Forest (GPU)")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {len(class_names)}")
    
    # Convert data to cupy arrays for GPU processing
    print("\nConverting data to GPU arrays...")
    X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
    y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
    X_test_gpu = cp.asarray(X_test, dtype=cp.float32)
    y_test_gpu = cp.asarray(y_test, dtype=cp.int32)
    
    # Print GPU info (correct method to get GPU name)
    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    print(f"GPU: {props['name'].decode()}")
    print(f"GPU Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
    
    # Default config or use provided
    # Convert sklearn parameters to cuML format
    if config is None:
        config = {
            'n_estimators': 500,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'split_criterion': 0,  # 0=GINI, 1=ENTROPY (cuML uses int)
            'random_state': 42,
            'n_bins': 128  # cuML-specific parameter
        }
    else:
        # Convert from Optuna config (which may have sklearn parameters)
        cuml_config = {}
        
        # Direct mappings
        for key in ['n_estimators', 'max_depth', 'min_samples_split', 
                    'min_samples_leaf', 'bootstrap', 'random_state']:
            if key in config:
                cuml_config[key] = config[key]
        
        # Convert criterion to split_criterion (int)
        if 'criterion' in config:
            cuml_config['split_criterion'] = 0 if config['criterion'] == 'gini' else 1
        elif 'split_criterion' in config:
            cuml_config['split_criterion'] = config['split_criterion']
        else:
            cuml_config['split_criterion'] = 0  # default to GINI
        
        # Handle max_features
        if 'max_features' in config:
            if config['max_features'] is None:
                cuml_config['max_features'] = 1.0  # cuML doesn't support None
            else:
                cuml_config['max_features'] = config['max_features']
        else:
            cuml_config['max_features'] = 'sqrt'
        
        # Add cuML-specific parameters
        cuml_config['n_bins'] = config.get('n_bins', 128)
        
        # Add max_samples if bootstrap is True
        if cuml_config.get('bootstrap', True) and 'max_samples' in config:
            cuml_config['max_samples'] = config['max_samples']
        
        # Remove sklearn-only parameters that cuML doesn't support
        # (n_jobs is not needed for cuML as it uses GPU)
        
        config = cuml_config
    
    print(f"\nRandom Forest config (GPU):")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Train model
    print("\nTraining Random Forest on GPU...")
    model = RandomForestClassifier(**config)
    
    model.fit(X_train_gpu, y_train_gpu)
    
    # Predictions
    print("  Making predictions on GPU...")
    y_pred_gpu = model.predict(X_test_gpu)
    y_pred_proba_gpu = model.predict_proba(X_test_gpu)
    
    # Convert predictions back to CPU for metrics and saving
    y_pred = cp.asnumpy(y_pred_gpu)
    y_pred_proba = cp.asnumpy(y_pred_proba_gpu)
    y_test_cpu = cp.asnumpy(y_test_gpu)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_cpu, y_pred)
    precision = precision_score(y_test_cpu, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_cpu, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_cpu, y_pred, average='weighted', zero_division=0)
    
    print(f"✓ Random Forest Accuracy: {accuracy:.4f}")
    
    # Print metrics
    print("\n" + "─"*70)
    print("FINAL Random Forest METRICS")
    print("─"*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test_cpu, y_pred)
    
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)
    
    # Save confusion matrices
    prefix = f"{classifier_name}_rf"
    
    # Raw counts
    plot_confusion_matrix(
        cm, class_names,
        f'{classifier_name.title()} Classification - Confusion Matrix (Random Forest GPU)',
        os.path.join(output_dir, f'{prefix}_confusion_matrix_raw.png'),
        normalize=False
    )
    
    # Normalized
    plot_confusion_matrix(
        cm, class_names,
        f'{classifier_name.title()} Classification - Confusion Matrix Normalized (Random Forest GPU)',
        os.path.join(output_dir, f'{prefix}_confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # Save CSV
    save_confusion_matrix_csv(
        cm, class_names,
        os.path.join(output_dir, f'{prefix}_confusion_matrix.csv')
    )
    
    print("─"*70)
    
    # Clean up GPU memory
    del X_train_gpu, y_train_gpu, X_test_gpu, y_test_gpu, y_pred_gpu, y_pred_proba_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    # Return results
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm
    }


# ============================================================================
# SECTION 5: Save Models
# ============================================================================

def save_models(plant_xgb_results, disease_xgb_results,
                plant_rf_results, disease_rf_results,
                plant_encoder, disease_encoder, 
                n_features, output_dir='.'):
    """Save all trained models."""
    
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save XGBoost models
    xgb_package = {
        'plant_model': plant_xgb_results['model'],
        'plant_encoder': plant_encoder,
        'plant_metrics': {
            'accuracy': plant_xgb_results['accuracy'],
            'precision': plant_xgb_results['precision'],
            'recall': plant_xgb_results['recall'],
            'f1_score': plant_xgb_results['f1_score']
        },
        'disease_model': disease_xgb_results['model'],
        'disease_encoder': disease_encoder,
        'disease_metrics': {
            'accuracy': disease_xgb_results['accuracy'],
            'precision': disease_xgb_results['precision'],
            'recall': disease_xgb_results['recall'],
            'f1_score': disease_xgb_results['f1_score']
        },
        'n_features': n_features,
        'model_type': 'xgboost',
        'timestamp': timestamp
    }
    
    xgb_path = os.path.join(output_dir, 'plant_disease_classifier_xgboost.pkl')
    joblib.dump(xgb_package, xgb_path)
    print(f"\n✓ XGBoost models saved to {xgb_path}")
    print(f"  Plant classifier: {plant_xgb_results['accuracy']:.4f}")
    print(f"  Disease classifier: {disease_xgb_results['accuracy']:.4f}")
    
    # Save Random Forest models
    rf_package = {
        'plant_model': plant_rf_results['model'],
        'plant_encoder': plant_encoder,
        'plant_metrics': {
            'accuracy': plant_rf_results['accuracy'],
            'precision': plant_rf_results['precision'],
            'recall': plant_rf_results['recall'],
            'f1_score': plant_rf_results['f1_score']
        },
        'disease_model': disease_rf_results['model'],
        'disease_encoder': disease_encoder,
        'disease_metrics': {
            'accuracy': disease_rf_results['accuracy'],
            'precision': disease_rf_results['precision'],
            'recall': disease_rf_results['recall'],
            'f1_score': disease_rf_results['f1_score']
        },
        'n_features': n_features,
        'model_type': 'random_forest',
        'timestamp': timestamp
    }
    
    rf_path = os.path.join(output_dir, 'plant_disease_classifier_rf.pkl')
    joblib.dump(rf_package, rf_path)
    print(f"\n✓ Random Forest models saved to {rf_path}")
    print(f"  Plant classifier: {plant_rf_results['accuracy']:.4f}")
    print(f"  Disease classifier: {disease_rf_results['accuracy']:.4f}")
    
    # Save label encoders separately
    encoder_path = os.path.join(output_dir, 'label_encoders_ml.pkl')
    encoder_package = {
        'plant_encoder': plant_encoder,
        'disease_encoder': disease_encoder,
        'n_features': n_features,
        'timestamp': timestamp
    }
    joblib.dump(encoder_package, encoder_path)
    print(f"\n✓ Label encoders saved to {encoder_path}")
    
    print("\n" + "="*70)
    print("MODEL EXPORT SUMMARY")
    print("="*70)
    print(f"✓ XGBoost package: plant_disease_classifier_xgboost.pkl")
    print(f"✓ Random Forest package: plant_disease_classifier_rf.pkl")
    print(f"✓ Label encoders: label_encoders_ml.pkl")
    print(f"✓ Confusion matrices (PNG & CSV) saved with prefixes:")
    print(f"  - plant_xgboost_*")
    print(f"  - disease_xgboost_*")
    print(f"  - plant_rf_*")
    print(f"  - disease_rf_*")


# ============================================================================
# SECTION 6: Main Training Pipeline
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train XGBoost and Random Forest classifiers from extracted features'
    )
    parser.add_argument('--train_features', type=str, default='plant_features_train.h5',
                       help='Path to training features HDF5 file')
    parser.add_argument('--test_features', type=str, default='plant_features_valid.h5',
                       help='Path to test/validation features HDF5 file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save trained models')
    parser.add_argument('--xgb_config', type=str, default=None,
                       help='Path to XGBoost config JSON (from Optuna)')
    parser.add_argument('--rf_config', type=str, default=None,
                       help='Path to Random Forest config JSON (from Optuna)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PLANT DISEASE CLASSIFICATION TRAINING")
    print("XGBoost (CPU) and Random Forest (GPU)")
    print("="*70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configs if provided
    xgb_plant_config = None
    xgb_disease_config = None
    rf_plant_config = None
    rf_disease_config = None
    
    if args.xgb_config:
        import json
        with open(args.xgb_config, 'r') as f:
            xgb_configs = json.load(f)
            xgb_plant_config = xgb_configs.get('xgboost_plant', {}).get('best_params')
            xgb_disease_config = xgb_configs.get('xgboost_disease', {}).get('best_params')
        print(f"\n✓ Loaded XGBoost configs from {args.xgb_config}")
    
    if args.rf_config:
        import json
        with open(args.rf_config, 'r') as f:
            rf_configs = json.load(f)
            rf_plant_config = rf_configs.get('rf_plant', {}).get('best_params')
            rf_disease_config = rf_configs.get('rf_disease', {}).get('best_params')
        print(f"✓ Loaded Random Forest configs from {args.rf_config}")
    
    # Step 1: Load training features
    print("\n" + "="*70)
    print("Step 1: Loading training features...")
    print("="*70)
    X_train, train_plants, train_diseases = load_features_hdf5(args.train_features)
    
    # Step 2: Load test features
    print("\n" + "="*70)
    print("Step 2: Loading test features...")
    print("="*70)
    X_test, test_plants, test_diseases = load_features_hdf5(args.test_features)
    
    # Step 3: Encode labels
    print("\n" + "="*70)
    print("Step 3: Encoding labels...")
    print("="*70)
    
    plant_encoder = LabelEncoder().fit(train_plants)
    disease_encoder = LabelEncoder().fit(train_diseases)
    
    y_train_plant = plant_encoder.transform(train_plants)
    y_test_plant = plant_encoder.transform(test_plants)
    
    y_train_disease = disease_encoder.transform(train_diseases)
    y_test_disease = disease_encoder.transform(test_diseases)
    
    print(f"\nPlants: {len(plant_encoder.classes_)} classes")
    print(f"  {plant_encoder.classes_}")
    print(f"Diseases: {len(disease_encoder.classes_)} classes")
    print(f"  {disease_encoder.classes_}")
    
    # Step 4: Train XGBoost plant classifier
    print("\n" + "="*70)
    print("TRAINING XGBoost PLANT CLASSIFIER")
    print("="*70)
    plant_xgb_results = train_xgboost_classifier(
        X_train, y_train_plant,
        X_test, y_test_plant,
        class_names=plant_encoder.classes_,
        classifier_name='plant',
        output_dir=args.output_dir,
        config=xgb_plant_config
    )
    
    # Step 5: Train XGBoost disease classifier
    print("\n" + "="*70)
    print("TRAINING XGBoost DISEASE CLASSIFIER")
    print("="*70)
    disease_xgb_results = train_xgboost_classifier(
        X_train, y_train_disease,
        X_test, y_test_disease,
        class_names=disease_encoder.classes_,
        classifier_name='disease',
        output_dir=args.output_dir,
        config=xgb_disease_config
    )
    
    # Step 6: Train Random Forest plant classifier
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST PLANT CLASSIFIER (GPU)")
    print("="*70)
    plant_rf_results = train_random_forest_classifier(
        X_train, y_train_plant,
        X_test, y_test_plant,
        class_names=plant_encoder.classes_,
        classifier_name='plant',
        output_dir=args.output_dir,
        config=rf_plant_config
    )
    
    # Step 7: Train Random Forest disease classifier
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST DISEASE CLASSIFIER (GPU)")
    print("="*70)
    disease_rf_results = train_random_forest_classifier(
        X_train, y_train_disease,
        X_test, y_test_disease,
        class_names=disease_encoder.classes_,
        classifier_name='disease',
        output_dir=args.output_dir,
        config=rf_disease_config
    )
    
    # Step 8: Save models
    save_models(
        plant_xgb_results, disease_xgb_results,
        plant_rf_results, disease_rf_results,
        plant_encoder, disease_encoder,
        n_features=X_train.shape[1],
        output_dir=args.output_dir
    )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nFinal Results:")
    print(f"\nXGBoost (CPU):")
    print(f"  Plant Classifier:   {plant_xgb_results['accuracy']:.4f}")
    print(f"  Disease Classifier: {disease_xgb_results['accuracy']:.4f}")
    print(f"\nRandom Forest (GPU):")
    print(f"  Plant Classifier:   {plant_rf_results['accuracy']:.4f}")
    print(f"  Disease Classifier: {disease_rf_results['accuracy']:.4f}")
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
