# -*- coding: utf-8 -*-
"""
Plant Disease Classification Inference - NODE Only
Supports both PKL and ONNX model formats.
"""

import os
import cv2
import numpy as np
import joblib
import argparse
from skimage.feature import local_binary_pattern


# ============================================================================
# SECTION 1: Leaf Segmenter (Optional)
# ============================================================================

class LeafSegmenter:
    """Loads and uses the pre-trained leaf segmenter"""
    
    def __init__(self, model_path='leaf_segmenter_multimodal.pkl'):
        print(f"Loading leaf segmenter from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.patch_size = model_data.get('patch_size', 64)
        print("✓ Leaf segmenter loaded")
    
    def extract_features(self, patch):
        """Extract features from patch"""
        return self.feature_extractor.extract_patch_features(patch)
    
    def segment(self, image, stride=None, threshold=0.5):
        """Segment leaf from image, return probability map"""
        if stride is None:
            stride = self.patch_size // 2
        
        h, w = image.shape[:2]
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    continue
                
                features = self.extract_features(patch).reshape(1, -1)
                try:
                    prob = self.model.predict_proba(features)[0][1]
                except:
                    prob = float(self.model.predict(features)[0])
                
                prob_map[y:y+self.patch_size, x:x+self.patch_size] += prob
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        prob_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), 
                            where=count_map > 0)
        mask = (prob_map > threshold).astype(np.uint8) * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, prob_map
    
    def extract_leaf(self, image, mask):
        """Extract leaf region with white background"""
        result = np.ones_like(image) * 255
        result[mask > 0] = image[mask > 0]
        return result


# ============================================================================
# SECTION 2: Feature Extraction
# ============================================================================

def extract_lbp_features(gray_image, num_points=8, radius=1):
    """Extract Local Binary Pattern features"""
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    n_bins = num_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_classification_features(image, segmenter=None, mask=None):
    """
    Extract comprehensive features for plant/disease classification.
    Total: 29 features (Color BGR: 6, Color HSV: 6, Sobel: 5, Laplacian: 2, LBP: 10)
    """
    
    # Use segmented leaf if available
    if segmenter is not None and mask is None:
        try:
            mask, _ = segmenter.segment(image)
            leaf = segmenter.extract_leaf(image, mask)
        except:
            leaf = image
    elif mask is not None:
        leaf = segmenter.extract_leaf(image, mask) if segmenter else image
    else:
        leaf = image
    
    # Resize for consistency
    leaf = cv2.resize(leaf, (256, 256))
    
    features = []
    
    # 1. Color features (BGR) - 6 features
    for channel in range(3):
        features.extend([np.mean(leaf[:,:,channel]), np.std(leaf[:,:,channel])])
    
    # 2. Color features (HSV) - 6 features
    hsv = cv2.cvtColor(leaf, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        features.extend([np.mean(hsv[:,:,channel]), np.std(hsv[:,:,channel])])
    
    # Convert to grayscale
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
    
    # 3. Texture features (Sobel) - 5 features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    features.extend([
        np.mean(np.abs(sobelx)), np.std(sobelx),
        np.mean(np.abs(sobely)), np.std(sobely),
        np.mean(np.sqrt(sobelx**2 + sobely**2))
    ])
    
    # 4. Laplacian - 2 features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
    
    # 5. LBP features - 10 features
    lbp_hist = extract_lbp_features(gray, num_points=8, radius=1)
    features.extend(lbp_hist)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# SECTION 3: PKL Classifier (NODE Only)
# ============================================================================

class PlantDiseaseClassifierPKL:
    """Inference using PKL models (NODE only)"""
    
    def __init__(self, model_path='plant_disease_classifier_node.pkl', 
                 segmenter_path='leaf_segmenter_multimodal.pkl'):
        
        print("Loading NODE classifier (PKL)...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at: {model_path}")
        
        self.models = joblib.load(model_path)
        self.plant_encoder = self.models['plant_encoder']
        self.disease_encoder = self.models['disease_encoder']
        self.plant_node = self.models['plant_node']
        self.disease_node = self.models['disease_node']
        
        print(f"  Model type: NODE")
        print(f"  Plant classes: {len(self.plant_encoder.classes_)}")
        print(f"  Disease classes: {len(self.disease_encoder.classes_)}")
        
        # Load segmenter
        self.segmenter = None
        self.use_segmentation = True
        
        if os.path.exists(segmenter_path):
            try:
                self.segmenter = LeafSegmenter(segmenter_path)
            except Exception as e:
                print(f"Warning: Could not load segmenter: {e}")
                self.use_segmentation = False
        else:
            print(f"Warning: Segmenter not found at {segmenter_path}")
            self.use_segmentation = False
        
        print("✓ NODE Classifier ready")
    
    def _predict_with_node(self, features, node_model):
        """Helper to predict using NODE model"""
        import pandas as pd
        
        feature_cols = [f'feature_{i}' for i in range(features.shape[1])]
        test_df = pd.DataFrame(features, columns=feature_cols)
        test_df['target'] = 0  # Dummy target
        
        # Get probabilities
        proba_df = node_model.predict(test_df, ret_logits=False)
        
        # Find probability columns
        proba_cols = [col for col in proba_df.columns 
                     if '_probability' in col.lower() or col.startswith('0') or col.startswith('1')]
        if len(proba_cols) == 0:
            exclude_cols = ['target', 'prediction'] + feature_cols
            proba_cols = [col for col in proba_df.columns if col not in exclude_cols]
        
        proba = proba_df[proba_cols].values
        return proba
    
    def predict(self, image_path):
        """Predict plant type and disease"""
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract features
        if self.use_segmentation and self.segmenter:
            mask, _ = self.segmenter.segment(img)
            features = extract_classification_features(img, self.segmenter, mask)
        else:
            features = extract_classification_features(img)
        
        features = features.reshape(1, -1)
        
        # Predict plant
        plant_proba = self._predict_with_node(features, self.plant_node)
        plant_pred = np.argmax(plant_proba[0])
        
        # Predict disease
        disease_proba = self._predict_with_node(features, self.disease_node)
        disease_pred = np.argmax(disease_proba[0])
        
        return {
            'plant': self.plant_encoder.classes_[plant_pred],
            'plant_confidence': float(plant_proba[0][plant_pred]),
            'disease': self.disease_encoder.classes_[disease_pred],
            'disease_confidence': float(disease_proba[0][disease_pred])
        }


# ============================================================================
# SECTION 4: ONNX Classifier (NODE Only)
# ============================================================================

class PlantDiseaseClassifierONNX:
    """Inference using ONNX models (NODE only)"""
    
    def __init__(self, encoder_path='label_encoders.pkl', 
                 segmenter_path='leaf_segmenter_multimodal.pkl',
                 model_dir='.'):
        
        print("Loading NODE classifier (ONNX)...")
        
        import onnxruntime as ort
        
        # Load encoders
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        encoders = joblib.load(encoder_path)
        self.plant_encoder = encoders['plant_encoder']
        self.disease_encoder = encoders['disease_encoder']
        
        print(f"  Model type: NODE")
        print(f"  Plant classes: {len(self.plant_encoder.classes_)}")
        print(f"  Disease classes: {len(self.disease_encoder.classes_)}")
        
        # Load ONNX models
        plant_onnx = os.path.join(model_dir, 'plant_classifier_node.onnx')
        disease_onnx = os.path.join(model_dir, 'disease_classifier_node.onnx')
        
        if not os.path.exists(plant_onnx):
            raise FileNotFoundError(f"Plant ONNX model not found: {plant_onnx}")
        if not os.path.exists(disease_onnx):
            raise FileNotFoundError(f"Disease ONNX model not found: {disease_onnx}")
        
        self.plant_session = ort.InferenceSession(plant_onnx)
        self.disease_session = ort.InferenceSession(disease_onnx)
        
        print(f"  ✓ Loaded plant ONNX: {os.path.basename(plant_onnx)}")
        print(f"  ✓ Loaded disease ONNX: {os.path.basename(disease_onnx)}")
        
        # Load segmenter
        self.segmenter = None
        self.use_segmentation = True
        
        if os.path.exists(segmenter_path):
            try:
                self.segmenter = LeafSegmenter(segmenter_path)
            except Exception as e:
                print(f"Warning: Could not load segmenter: {e}")
                self.use_segmentation = False
        else:
            print(f"Warning: Segmenter not found at {segmenter_path}")
            self.use_segmentation = False
        
        print("✓ ONNX Classifier ready")
    
    def predict(self, image_path):
        """Predict plant type and disease using ONNX"""
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract features
        if self.use_segmentation and self.segmenter:
            mask, _ = self.segmenter.segment(img)
            features = extract_classification_features(img, self.segmenter, mask)
        else:
            features = extract_classification_features(img)
        
        features = features.reshape(1, -1).astype(np.float32)
        
        # Predict plant
        plant_input_name = self.plant_session.get_inputs()[0].name
        plant_outputs = self.plant_session.run(None, {plant_input_name: features})
        
        # Handle different output formats
        if len(plant_outputs) > 1:
            plant_proba = plant_outputs[1]  # Probabilities usually second output
        else:
            plant_proba = plant_outputs[0]
        
        # Ensure probabilities are in correct format
        if len(plant_proba.shape) == 1:
            plant_proba = plant_proba.reshape(1, -1)
        
        plant_pred = np.argmax(plant_proba[0])
        
        # Predict disease
        disease_input_name = self.disease_session.get_inputs()[0].name
        disease_outputs = self.disease_session.run(None, {disease_input_name: features})
        
        if len(disease_outputs) > 1:
            disease_proba = disease_outputs[1]
        else:
            disease_proba = disease_outputs[0]
        
        if len(disease_proba.shape) == 1:
            disease_proba = disease_proba.reshape(1, -1)
        
        disease_pred = np.argmax(disease_proba[0])
        
        return {
            'plant': self.plant_encoder.classes_[plant_pred],
            'plant_confidence': float(plant_proba[0][plant_pred]),
            'disease': self.disease_encoder.classes_[disease_pred],
            'disease_confidence': float(disease_proba[0][disease_pred])
        }


# ============================================================================
# SECTION 5: Main Inference
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Classification Inference (NODE only)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--format', type=str, choices=['pkl', 'onnx'], default='pkl',
                       help='Model format to use (pkl or onnx)')
    parser.add_argument('--model_path', type=str, default='plant_disease_classifier_node.pkl',
                       help='Path to PKL model')
    parser.add_argument('--encoder_path', type=str, default='label_encoders.pkl',
                       help='Path to label encoders (for ONNX)')
    parser.add_argument('--segmenter_path', type=str, default='leaf_segmenter_multimodal.pkl',
                       help='Path to leaf segmenter model')
    parser.add_argument('--model_dir', type=str, default='.',
                       help='Directory containing ONNX models')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    print("="*70)
    print("PLANT DISEASE CLASSIFICATION INFERENCE - NODE")
    print("="*70)
    print(f"Image: {args.image}")
    print(f"Format: {args.format.upper()}")
    print()
    
    # Load classifier
    try:
        if args.format == 'pkl':
            classifier = PlantDiseaseClassifierPKL(
                model_path=args.model_path,
                segmenter_path=args.segmenter_path
            )
        else:  # onnx
            classifier = PlantDiseaseClassifierONNX(
                encoder_path=args.encoder_path,
                segmenter_path=args.segmenter_path,
                model_dir=args.model_dir
            )
        
        # Make prediction
        print("\nMaking prediction...")
        result = classifier.predict(args.image)
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\n🌿 Plant Type: {result['plant']}")
        print(f"   Confidence: {result['plant_confidence']:.2%}")
        print(f"   Model: NODE")
        
        print(f"\n🦠 Disease: {result['disease']}")
        print(f"   Confidence: {result['disease_confidence']:.2%}")
        print(f"   Model: NODE")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()