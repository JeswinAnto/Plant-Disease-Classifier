# -*- coding: utf-8 -*-
"""
Plant Disease Feature Extraction Pipeline
Extracts features including color, texture, gradients, and LBP from plant images.
Saves features to HDF5 format for efficient storage and loading.
"""

import os
import cv2
import numpy as np
import joblib
import h5py
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import argparse


# ============================================================================
# SECTION 1: Leaf Segmenter (Loads pre-trained model)
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
# SECTION 2: Feature Extraction with LBP
# ============================================================================

def extract_lbp_features(gray_image, num_points=8, radius=1):
    """
    Extract Local Binary Pattern features
    
    Args:
        gray_image: Grayscale image
        num_points: Number of circularly symmetric neighbor points (default: 8)
        radius: Radius of circle (default: 1)
    
    Returns:
        LBP histogram features
    """
    # Compute LBP
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    
    # Calculate histogram
    n_bins = num_points + 2  # uniform LBP has num_points + 2 bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_classification_features(image, segmenter=None, mask=None):
    """
    Extract comprehensive features for plant/disease classification.
    Includes: Color (BGR, HSV), Texture (Sobel, Laplacian), and LBP features.
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
    
    # Convert to grayscale for texture features
    gray = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
    
    # 3. Texture features (Sobel gradients) - 5 features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    features.extend([
        np.mean(np.abs(sobelx)), np.std(sobelx),
        np.mean(np.abs(sobely)), np.std(sobely),
        np.mean(np.sqrt(sobelx**2 + sobely**2))
    ])
    
    # 4. Laplacian (second derivative) - 2 features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
    
    # 5. LBP features - 10 features (8 neighbors + 2 for uniform patterns)
    lbp_hist = extract_lbp_features(gray, num_points=8, radius=1)
    features.extend(lbp_hist)
    
    return np.array(features, dtype=np.float32)


# ============================================================================
# SECTION 3: Dataset Loading
# ============================================================================

def load_plant_dataset(dataset_root='/home/jeswin/dataset/Plant Dataset', 
                       split='train', max_images_per_class=None):
    """
    Load images from the Plant Dataset folder structure.
    Expected structure: {split}/{plant___disease}/{image}.jpg
    """
    
    split_path = os.path.join(dataset_root, split)
    if not os.path.exists(split_path):
        raise ValueError(f"Split path not found: {split_path}")
    
    print(f"\nLoading {split} dataset from {split_path}...")
    
    images = []
    plant_labels = []
    disease_labels = []
    image_paths = []
    
    # Get all class folders
    class_folders = [d for d in os.listdir(split_path) 
                     if os.path.isdir(os.path.join(split_path, d))]
    
    for class_folder in sorted(class_folders):
        class_path = os.path.join(split_path, class_folder)
        
        # Parse plant and disease from folder name
        if '___' in class_folder:
            plant, disease = class_folder.split('___', 1)
        else:
            plant = class_folder
            disease = 'unknown'
        
        # Load images from this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        count = 0
        for img_file in image_files:
            if max_images_per_class and count >= max_images_per_class:
                break
            
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                images.append(img)
                plant_labels.append(plant)
                disease_labels.append(disease)
                image_paths.append(img_path)
                count += 1
        
        print(f"  {class_folder}: {count} images")
    
    print(f"Total images loaded: {len(images)}")
    return images, plant_labels, disease_labels, image_paths


# ============================================================================
# SECTION 4: Feature Extraction Pipeline
# ============================================================================

def extract_all_features(images, segmenter=None, use_segmentation=True):
    """Extract features from all images with progress bar."""
    
    print("\nExtracting features...")
    features = []
    failed_indices = []
    
    for i, img in enumerate(tqdm(images, desc="Processing images")):
        try:
            if use_segmentation and segmenter is not None:
                mask, _ = segmenter.segment(img)
                feat = extract_classification_features(img, segmenter, mask)
            else:
                feat = extract_classification_features(img)
            
            features.append(feat)
        except Exception as e:
            print(f"\nError processing image {i}: {e}")
            # Use zero features as placeholder
            features.append(np.zeros(29))  # 6+6+5+2+10 = 29 features
            failed_indices.append(i)
    
    features = np.array(features, dtype=np.float32)
    print(f"Feature shape: {features.shape}")
    
    if failed_indices:
        print(f"Warning: {len(failed_indices)} images failed processing")
    
    return features, failed_indices


# ============================================================================
# SECTION 5: Save Features to HDF5
# ============================================================================

def save_features_hdf5(features, plant_labels, disease_labels, image_paths, 
                       output_path, split_name, failed_indices=None):
    """
    Save extracted features and metadata to HDF5 file.
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        plant_labels: list of plant type labels
        disease_labels: list of disease labels
        image_paths: list of image file paths
        output_path: path to save HDF5 file
        split_name: 'train' or 'valid'
        failed_indices: list of indices that failed feature extraction
    """
    
    print(f"\nSaving features to {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Save features
        f.create_dataset('features', data=features, compression='gzip', compression_opts=9)
        
        # Save labels (encode as bytes for HDF5 compatibility)
        dt = h5py.special_dtype(vlen=str)
        plant_labels_array = np.array(plant_labels, dtype=object)
        disease_labels_array = np.array(disease_labels, dtype=object)
        image_paths_array = np.array(image_paths, dtype=object)
        
        f.create_dataset('plant_labels', data=plant_labels_array, dtype=dt)
        f.create_dataset('disease_labels', data=disease_labels_array, dtype=dt)
        f.create_dataset('image_paths', data=image_paths_array, dtype=dt)
        
        # Save metadata
        f.attrs['split'] = split_name
        f.attrs['n_samples'] = len(features)
        f.attrs['n_features'] = features.shape[1]
        f.attrs['n_plants'] = len(set(plant_labels))
        f.attrs['n_diseases'] = len(set(disease_labels))
        
        if failed_indices:
            f.create_dataset('failed_indices', data=np.array(failed_indices))
        
        # Save unique class names
        unique_plants = sorted(list(set(plant_labels)))
        unique_diseases = sorted(list(set(disease_labels)))
        
        unique_plants_array = np.array(unique_plants, dtype=object)
        unique_diseases_array = np.array(unique_diseases, dtype=object)
        
        f.create_dataset('unique_plants', data=unique_plants_array, dtype=dt)
        f.create_dataset('unique_diseases', data=unique_diseases_array, dtype=dt)
    
    print(f"✓ Features saved successfully")
    print(f"  Samples: {len(features)}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Plants: {len(set(plant_labels))}")
    print(f"  Diseases: {len(set(disease_labels))}")


# ============================================================================
# SECTION 6: Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract features from plant disease dataset')
    parser.add_argument('--dataset_root', type=str, 
                       default='/home/jeswin/dataset/Plant Dataset',
                       help='Root directory of Plant Dataset')
    parser.add_argument('--segmenter_path', type=str, 
                       default='leaf_segmenter_multimodal.pkl',
                       help='Path to leaf segmenter model')
    parser.add_argument('--output_dir', type=str, 
                       default='.',
                       help='Directory to save extracted features')
    parser.add_argument('--max_images_per_class', type=int, 
                       default=None,
                       help='Maximum images per class (None for all)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PLANT DISEASE FEATURE EXTRACTION PIPELINE")
    print("Features: Color (BGR, HSV) + Texture (Sobel, Laplacian) + LBP")
    print("="*70)
    
    # Step 1: Load segmenter (if available)
    print("\nStep 1: Loading leaf segmenter...")
    segmenter = None
    use_segmentation = False
    
    try:
        if os.path.exists(args.segmenter_path):
            segmenter = LeafSegmenter(args.segmenter_path)
            use_segmentation = True
            print("✓ Segmenter loaded - will use segmentation")
        else:
            print(f"⚠ Segmenter not found at {args.segmenter_path}")
            print("  Continuing without segmentation...")
    except Exception as e:
        print(f"⚠ Could not load segmenter: {e}")
        print("  Continuing without segmentation...")
    
    # Step 2: Process training set
    print("\n" + "="*70)
    print("PROCESSING TRAINING SET")
    print("="*70)
    
    train_images, train_plants, train_diseases, train_paths = load_plant_dataset(
        dataset_root=args.dataset_root,
        split='train',
        max_images_per_class=args.max_images_per_class
    )
    
    if len(train_images) == 0:
        print("✗ No training images found!")
        return
    
    train_features, train_failed = extract_all_features(
        train_images, 
        segmenter=segmenter,
        use_segmentation=use_segmentation
    )
    
    train_output = os.path.join(args.output_dir, 'plant_features_train.h5')
    save_features_hdf5(
        train_features, train_plants, train_diseases, train_paths,
        train_output, 'train', train_failed
    )
    
    # Step 3: Process validation set
    print("\n" + "="*70)
    print("PROCESSING VALIDATION SET")
    print("="*70)
    
    try:
        valid_images, valid_plants, valid_diseases, valid_paths = load_plant_dataset(
            dataset_root=args.dataset_root,
            split='valid',
            max_images_per_class=args.max_images_per_class
        )
        
        if len(valid_images) == 0:
            print("⚠ No validation images found!")
        else:
            valid_features, valid_failed = extract_all_features(
                valid_images,
                segmenter=segmenter,
                use_segmentation=use_segmentation
            )
            
            valid_output = os.path.join(args.output_dir, 'plant_features_valid.h5')
            save_features_hdf5(
                valid_features, valid_plants, valid_diseases, valid_paths,
                valid_output, 'valid', valid_failed
            )
    
    except Exception as e:
        print(f"⚠ Could not process validation set: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print(f"✓ Training features saved to: {train_output}")
    if os.path.exists(os.path.join(args.output_dir, 'plant_features_valid.h5')):
        print(f"✓ Validation features saved to: {valid_output}")
    print(f"\nFeature dimensions: {train_features.shape[1]}")
    print("  - Color (BGR): 6 features")
    print("  - Color (HSV): 6 features")
    print("  - Texture (Sobel): 5 features")
    print("  - Texture (Laplacian): 2 features")
    print("  - LBP (uniform): 10 features")
    print("  - Total: 29 features")


if __name__ == "__main__":
    main()