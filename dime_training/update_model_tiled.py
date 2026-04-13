#!/usr/bin/env python3
"""
Script to update trained model with new/missing frames with tiling and DDP support
"""

import os
import torch
import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_health(model_path):
    """Check if model files exist and are valid"""
    required_files = [
        "dime_params.pkl",
        "nnscorer_search_index.faiss", 
        "anomaly_rater_features.pkl"  # This should exist now
    ]
    
    print(f"🔍 Checking model health in: {model_path}")
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {file}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {file}: MISSING")
            return False
    
    # Check feature count in the FAISS index
    try:
        from anomaly_engine.core.core_utils import ApproximateProximitySearcher
        nn_method = ApproximateProximitySearcher(False, 4)
        nn_method.load(os.path.join(model_path, "nnscorer_search_index.faiss"))
        feature_count = nn_method.search_index.ntotal
        print(f"   📊 FAISS index contains {feature_count} features")
        return feature_count > 0
    except Exception as e:
        print(f"   ❌ Failed to check FAISS index: {e}")
        return False

def update_tiled_model_with_frames(model_root_path, new_frames_dir, config, dataset_name="Area1"):
    """
    Update tiled model with new frames, handling all positions and ranks
    """
    print(f"🔄 Updating tiled model at {model_root_path} with frames from {new_frames_dir}")
    
    # Load the training config to get tiling parameters
    config_path = os.path.join(model_root_path, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        til_cfg = training_config.get("tiling", {})
        selector_config = training_config.get("selector", {})
        print(f"📋 Tiling config: {til_cfg}")
        print(f"🎯 Selector config: {selector_config}")
    else:
        print("⚠️ Warning: No training config found. Using default tiling.")
        til_cfg = {"enable": True, "rows": 2, "cols": 2, "overlap": 0.1}
        selector_config = {"name": "approx_greedy_coreset", "percentage": 0.01}
    
    # Check if this is a tiled model by looking for position directories
    dataset_model_path = os.path.join(model_root_path, "models", dataset_name)
    
    if not os.path.exists(dataset_model_path):
        print(f"❌ Dataset model path not found: {dataset_model_path}")
        return False
    
    # Look for rank directories
    rank_dirs = [d for d in os.listdir(dataset_model_path) if d.startswith("rank")]
    
    if not rank_dirs:
        print("❌ No rank directories found. This doesn't appear to be a tiled model.")
        return False
    
    print(f"📁 Found {len(rank_dirs)} rank directories")
    
    # Load existing model configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Create DataLoader for new frames
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from torchvision import transforms
    
    class NewFramesDataset(Dataset):
        def __init__(self, frames_dir, transform):
            self.frames_dir = Path(frames_dir)
            self.image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                self.image_paths.extend(list(self.frames_dir.rglob(ext)))
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            return {
                "image": image,
                "image_path": str(image_path),
                "image_name": image_path.name
            }
    
    transform = transforms.Compose([
        transforms.Resize(config["dataset"]["imagesize"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.601, 0.601, 0.601], 
                           std=[0.340, 0.340, 0.340]),
    ])
    
    new_dataset = NewFramesDataset(new_frames_dir, transform)
    new_dataloader = DataLoader(
        new_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"📸 Found {len(new_dataset)} new frames")
    
    if len(new_dataset) == 0:
        print("❌ No frames found in the specified directory")
        return False
    
    # Check model health before update
    print("\n🔍 Pre-update model health check:")
    all_healthy = True
    for rank_dir in rank_dirs:
        rank_path = os.path.join(dataset_model_path, rank_dir)
        pos_dirs = [d for d in os.listdir(rank_path) if d.startswith("pos")]
        
        for pos_dir in pos_dirs:
            pos_path = os.path.join(rank_path, pos_dir)
            if not check_model_health(pos_path):
                all_healthy = False
    
    if not all_healthy:
        print("❌ Some models are unhealthy. Please retrain the models first.")
        return False
    
    # Update each rank directory
    success_count = 0
    for rank_dir in rank_dirs:
        rank_path = os.path.join(dataset_model_path, rank_dir)
        print(f"\n🔄 Updating rank: {rank_dir}")
        
        # Find all position directories in this rank
        pos_dirs = [d for d in os.listdir(rank_path) if d.startswith("pos")]
        
        for pos_dir in pos_dirs:
            pos_path = os.path.join(rank_path, pos_dir)
            print(f"  🎯 Updating position: {pos_dir}")
            
            try:
                # Load the position-specific model
                from anomaly_engine.core.anomaly_net import AnomalyNet
                from anomaly_engine.core.core_utils import ApproximateProximitySearcher
                
                print(f"    📥 Loading model from {pos_path}")
                anomaly_net = AnomalyNet(device)
                anomaly_net.load_from_path(
                    pos_path, 
                    device, 
                    nn_method=ApproximateProximitySearcher(False, 4)
                )
                
                # Get current feature count
                current_count = anomaly_net.anomaly_rater.get_feature_count()
                print(f"    📊 Current features: {current_count}")
                
                # Update this position's model
                print("    🔄 Starting update with new frames...")
                success = anomaly_net.update_with_new_frames(
                    new_dataloader, 
                    update_percentage=0.5,
                    til_cfg=til_cfg,
                    is_ddp=False,  # Single process for update
                    selector_config=selector_config
                )
                
                if success:
                    # Save the updated model back to the same position directory
                    print("    💾 Saving updated model...")
                    anomaly_net.save_to_path(pos_path)
                    
                    # Verify the update worked
                    new_count = anomaly_net.anomaly_rater.get_feature_count()
                    added_count = new_count - current_count
                    
                    print(f"    ✅ Updated position {pos_dir}: +{added_count} features ({current_count} → {new_count})")
                    success_count += 1
                else:
                    print(f"    ❌ Failed to update position {pos_dir}")
                    
            except Exception as e:
                print(f"    💥 Error updating position {pos_dir}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n📊 Update Summary:")
    print(f"   ✅ Successfully updated {success_count} positions")
    print(f"   📁 Total positions: {sum(len([d for d in os.listdir(os.path.join(dataset_model_path, rd)) if d.startswith('pos')]) for rd in rank_dirs)}")
    
    # Update the main model index
    try:
        from train_tuple import build_models_index
        build_models_index(os.path.join(model_root_path, "models"), dataset_name)
        print("✅ Updated models index")
    except Exception as e:
        print(f"⚠️ Could not update models index: {e}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Update trained tiled model with new frames")
    parser.add_argument("--model_root", type=str, required=True, 
                       help="Root path to the trained model (contains training_config.json)")
    parser.add_argument("--new_frames", type=str, required=True,
                       help="Path to directory containing new frames")
    parser.add_argument("--dataset_name", type=str, default="Area1",
                       help="Dataset name for the model")
    
    args = parser.parse_args()
    
    # Configuration (should match your training config)
    config = {
        "dataset": {
            "imagesize": (707, 2259)  # Adjust based on your training
        }
    }
    
    success = update_tiled_model_with_frames(args.model_root, args.new_frames, config, args.dataset_name)
    
    if success:
        print("\n🎉 Model update completed successfully!")
    else:
        print("\n💥 Model update failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()