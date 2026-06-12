# -*- coding: utf-8 -*-
"""
Plant Disease Classification Training Pipeline - NODE Only
Trains NODE (Neural Oblivious Decision Ensemble) from pre-extracted features.
Saves models to ONNX format and generates confusion matrices.
"""

import os
import numpy as np
import h5py
# ============================================================================
# CRITICAL FIX: Patch torch.load BEFORE importing pytorch_tabular
# This must happen before any other imports that use torch.load
# ============================================================================
import torch

_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """Patched torch.load that handles omegaconf checkpoints in PyTorch 2.6+"""
    
    # If weights_only not explicitly set, try safe first then fall back
    if weights_only is None:
        try:
            return _original_torch_load(
                f, 
                map_location=map_location, 
                pickle_module=pickle_module, 
                weights_only=True, 
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            if "weights_only" in error_msg or "Unsupported global" in error_msg:
                # Fall back to unsafe loading for trusted checkpoints
                print(f"  → Using weights_only=False for checkpoint (trusted source)")
                
                # Import pickle for custom unpickler
                import pickle
                import io
                
                # Try with a custom unpickler that handles persistent IDs
                try:
                    # Read file content
                    if isinstance(f, str):
                        with open(f, 'rb') as file:
                            buffer = io.BytesIO(file.read())
                    elif hasattr(f, 'read'):
                        # It's already a file-like object
                        buffer = io.BytesIO(f.read())
                        if hasattr(f, 'seek'):
                            f.seek(0)  # Reset for potential retry
                    else:
                        buffer = f
                    
                    # Create custom unpickler that handles persistent references
                    class CustomUnpickler(pickle.Unpickler):
                        def persistent_load(self, pid):
                            # Handle persistent references - just return the pid
                            # PyTorch will handle the actual loading
                            return pid
                    
                    # Try to load with custom unpickler
                    unpickler = CustomUnpickler(buffer)
                    result = unpickler.load()
                    return result
                    
                except Exception as inner_e:
                    # If custom unpickler fails, try the standard unsafe load
                    # but with explicit mmap handling
                    print(f"  → Custom unpickler failed, trying standard unsafe load")
                    
                    # Reset file pointer if possible
                    if hasattr(f, 'seek'):
                        f.seek(0)
                    elif isinstance(f, str):
                        # Reopen the file
                        pass
                    
                    # Try with weights_only=False and mmap disabled
                    try:
                        return _original_torch_load(
                            f,
                            map_location=map_location,
                            weights_only=False,
                            mmap=False
                        )
                    except TypeError:
                        # If mmap parameter doesn't exist, try without it
                        return _original_torch_load(
                            f,
                            map_location=map_location,
                            weights_only=False
                        )
            else:
                raise
    else:
        # User explicitly set weights_only, pass everything through
        return _original_torch_load(
            f, 
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=weights_only, 
            **kwargs
        )

# Replace torch.load globally BEFORE any other imports
torch.load = patched_torch_load
print("✓ Patched torch.load for PyTorch 2.6+ compatibility\n")

# Now safe to import everything else
import pandas as pd
import gc
import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pytorch_tabular import TabularModel
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig


# ============================================================================
# SECTION 1: Load Features from HDF5
# ============================================================================

def load_features_hdf5(file_path):
    """Load features and labels from HDF5 file."""
    print(f"\nLoading features from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        plant_labels = [s for s in f['plant_labels'][:]]
        disease_labels = [s for s in f['disease_labels'][:]]
        
        metadata = {
            'split': f.attrs.get('split', 'unknown'),
            'n_samples': f.attrs.get('n_samples', len(features)),
            'n_features': f.attrs.get('n_features', features.shape[1]),
            'n_plants': f.attrs.get('n_plants', len(set(plant_labels))),
            'n_diseases': f.attrs.get('n_diseases', len(set(disease_labels)))
        }
        
        if 'failed_indices' in f:
            metadata['failed_indices'] = list(f['failed_indices'][:])
    
    print(f"✓ Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"  Plants: {metadata['n_plants']} classes")
    print(f"  Diseases: {metadata['n_diseases']} classes")
    
    return features, plant_labels, disease_labels, metadata


# ============================================================================
# SECTION 2: Confusion Matrix Visualization
# ============================================================================

def plot_confusion_matrix(cm, class_names, title, output_path, normalize=False):
    """Plot and save confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(max(10, len(class_names) * 0.8), max(8, len(class_names) * 0.6)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


# ============================================================================
# SECTION 3: NODE Model Training
# ============================================================================

def train_node_classifier(X_train, y_train, X_test, y_test, class_names,
                          classifier_name='plant', val_split=0.15, output_dir='.'):
    """Train NODE classifier and generate confusion matrix."""
    print(f"\n{'='*70}")
    print(f"Training {classifier_name.upper()} Classifier - NODE")
    print(f"{'='*70}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {len(np.unique(y_train))}")
    
    from sklearn.model_selection import train_test_split
    X_train_node, X_val_node, y_train_node, y_val_node = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
    )
    
    print("\nTraining NODE (Neural Oblivious Decision Ensemble)...")
    
    try:
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
        target_col = 'target'
        
        train_df = pd.DataFrame(X_train_node, columns=feature_cols)
        train_df[target_col] = y_train_node
        
        val_df = pd.DataFrame(X_val_node, columns=feature_cols)
        val_df[target_col] = y_val_node
        
        test_df = pd.DataFrame(X_test, columns=feature_cols)
        test_df[target_col] = y_test
        
        data_config = DataConfig(
            target=[target_col],
            continuous_cols=feature_cols,
            categorical_cols=[],
            num_workers=11
        )
        
        checkpoint_dir = os.path.join(output_dir, f'checkpoints_{classifier_name}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # ─── DYNAMIC CONFIGURATION ADJUSTMENT ───
        if classifier_name == 'disease':
            print("  → Applying optimized hyperparameters for DISEASE classifier...")
            batch_size = 256
            num_layers = 2
            num_trees = 84
            depth = 8
            learning_rate = 0.02999293839648055
        else:
            print("  → Applying default hyperparameters for PLANT classifier...")
            batch_size = 128
            num_layers = 2
            num_trees = 24
            depth = 8
            learning_rate = 0.01

        trainer_config = TrainerConfig(
            batch_size=batch_size,
            max_epochs=100,
            early_stopping_patience=20,
            early_stopping_min_delta=0.001,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            progress_bar='simple',
            checkpoints_path=checkpoint_dir,
            load_best=True
        )
        
        optimizer_config = OptimizerConfig()
        
        model_config = NodeConfig(
            task="classification",
            num_layers=num_layers,
            num_trees=num_trees,
            depth=depth,
            learning_rate=learning_rate
        )
        
        node_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )
        
        print("  Fitting NODE model...")
        node_model.fit(train=train_df, validation=val_df)
        
        print("  Making predictions...")
        node_proba_df = node_model.predict(test_df, ret_logits=False)
        
        node_proba_cols = [col for col in node_proba_df.columns 
                          if '_probability' in col.lower() or col.startswith('0') or col.startswith('1')]
        
        if len(node_proba_cols) == 0:
            exclude_cols = [target_col, 'prediction'] + feature_cols
            node_proba_cols = [col for col in node_proba_df.columns if col not in exclude_cols]
        
        print(f"  Probability columns used: {len(node_proba_cols)}")
        
        node_proba = node_proba_df[node_proba_cols].values
        node_pred = np.argmax(node_proba, axis=1)
        
        node_acc = accuracy_score(y_test, node_pred)
        print(f"✓ NODE Accuracy: {node_acc:.4f}")
        
    except Exception as e:
        print(f"✗ NODE training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    torch.cuda.empty_cache()
    gc.collect()
    
    average_type = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
    precision = precision_score(y_test, node_pred, average=average_type, zero_division=0)
    recall = recall_score(y_test, node_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_test, node_pred, average=average_type, zero_division=0)
    
    print(f"\n{'─'*70}")
    print(f"FINAL NODE METRICS")
    print(f"{'─'*70}")
    print(f"Accuracy:  {node_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, node_pred)
    
    cm_raw_path = os.path.join(output_dir, f'{classifier_name}_confusion_matrix_raw.png')
    cm_norm_path = os.path.join(output_dir, f'{classifier_name}_confusion_matrix_normalized.png')
    
    plot_confusion_matrix(cm, class_names, f'{classifier_name.capitalize()} Classification - Confusion Matrix (Raw)', cm_raw_path, normalize=False)
    plot_confusion_matrix(cm, class_names, f'{classifier_name.capitalize()} Classification - Confusion Matrix (Normalized)', cm_norm_path, normalize=True)
    
    cm_csv_path = os.path.join(output_dir, f'{classifier_name}_confusion_matrix.csv')
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv_path)
    print(f"✓ Confusion matrix CSV saved to {cm_csv_path}")
    
    print(f"{'─'*70}")
    
    return {
        'model': node_model, 'accuracy': node_acc, 'precision': precision, 'recall': recall,
        'f1_score': f1, 'predictions': node_pred, 'probabilities': node_proba, 'confusion_matrix': cm
    }


# ============================================================================
# SECTION 4: ONNX Export
# ============================================================================

class ONNXWrapper(torch.nn.Module):
    """Wrapper for pytorch_tabular models to enable ONNX export."""
    def __init__(self, tabular_model, n_features):
        super().__init__()
        self.model = tabular_model.model
        self.n_features = n_features
        
    def forward(self, x):
        x_dict = {
            'continuous': x,
            'categorical': torch.tensor([]).reshape(x.shape[0], 0)
        }
        return self.model(x_dict)


def export_node_to_onnx(node_model, n_features, output_path):
    """Export NODE model to ONNX format with proper input handling"""
    print(f"\nExporting NODE to ONNX...")
    try:
        wrapper = ONNXWrapper(node_model, n_features)
        wrapper.eval()
        dummy_input = torch.randn(1, n_features).float()
        device = next(wrapper.parameters()).device
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            test_output = wrapper(dummy_input)
            print(f"  Wrapper test passed. Output shape: {test_output.shape}")
        
        torch.onnx.export(
            wrapper, dummy_input, output_path, export_params=True, opset_version=12,
            do_constant_folding=True, input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, verbose=False
        )
        print(f"✓ NODE ONNX saved to {output_path}")
        
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX model verification passed")
        except ImportError:
            pass
        return True
    except Exception as e:
        print(f"✗ NODE ONNX export failed: {e}")
        return False


# ============================================================================
# SECTION 5: Save Models
# ============================================================================

def save_models(plant_results, disease_results, plant_encoder, disease_encoder, n_features, output_dir='.'):
    """Save all models in both PKL and ONNX formats."""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    pkl_path = os.path.join(output_dir, 'plant_disease_classifier_node.pkl')
    model_package = {
        'plant_node': plant_results['model'], 'plant_encoder': plant_encoder,
        'plant_metrics': {'accuracy': plant_results['accuracy'], 'precision': plant_results['precision'], 'recall': plant_results['recall'], 'f1_score': plant_results['f1_score']},
        'disease_node': disease_results['model'], 'disease_encoder': disease_encoder,
        'disease_metrics': {'accuracy': disease_results['accuracy'], 'precision': disease_results['precision'], 'recall': disease_results['recall'], 'f1_score': disease_results['f1_score']},
        'n_features': n_features, 'model_type': 'node'
    }
    joblib.dump(model_package, pkl_path)
    print(f"\n✓ PKL models saved to {pkl_path}")
    
    plant_node_onnx = os.path.join(output_dir, 'plant_classifier_node.onnx')
    export_node_to_onnx(plant_results['model'], n_features, plant_node_onnx)
    
    disease_node_onnx = os.path.join(output_dir, 'disease_classifier_node.onnx')
    export_node_to_onnx(disease_results['model'], n_features, disease_node_onnx)
    
    encoder_path = os.path.join(output_dir, 'label_encoders.pkl')
    encoder_package = {'plant_encoder': plant_encoder, 'disease_encoder': disease_encoder, 'n_features': n_features, 'model_type': 'node'}
    joblib.dump(encoder_package, encoder_path)
    print(f"✓ Label encoders saved to {encoder_path}")


# ============================================================================
# SECTION 6: Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train plant disease classifier (NODE only) from extracted features')
    parser.add_argument('--train_features', type=str, default='plant_features_train.h5', help='Path to training features HDF5 file')
    parser.add_argument('--test_features', type=str, default='plant_features_valid.h5', help='Path to test/validation features HDF5 file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save trained models')
    args = parser.parse_args()
    
    print("="*70)
    print("PLANT DISEASE CLASSIFICATION TRAINING PIPELINE")
    print("NODE (Neural Oblivious Decision Ensemble) - Only")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    X_train, train_plants, train_diseases, train_meta = load_features_hdf5(args.train_features)
    X_test, test_plants, test_diseases, test_meta = load_features_hdf5(args.test_features)
    
    plant_encoder = LabelEncoder().fit(train_plants)
    disease_encoder = LabelEncoder().fit(train_diseases)
    
    y_train_plant = plant_encoder.transform(train_plants)
    y_test_plant = plant_encoder.transform(test_plants)
    y_train_disease = disease_encoder.transform(train_diseases)
    y_test_disease = disease_encoder.transform(test_diseases)
    
    print("\n" + "="*70)
    print("TRAINING PLANT CLASSIFIER")
    print("="*70)
    plant_results = train_node_classifier(X_train, y_train_plant, X_test, y_test_plant, class_names=plant_encoder.classes_, classifier_name='plant', output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("TRAINING DISEASE CLASSIFIER")
    print("="*70)
    disease_results = train_node_classifier(X_train, y_train_disease, X_test, y_test_disease, class_names=disease_encoder.classes_, classifier_name='disease', output_dir=args.output_dir)
    
    save_models(plant_results, disease_results, plant_encoder, disease_encoder, n_features=X_train.shape[1], output_dir=args.output_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()