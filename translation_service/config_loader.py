"""
Configuration loader with YAML support
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PipelineConfig:
    """Enhanced configuration with YAML support."""
    
    # Image processing
    image_size: Tuple[int, int] = (512, 512)
    crop_size: Tuple[int, int] = (299, 299)
    gaussian_kernel: Tuple[int, int] = (5, 5)
    
    # SAM parameters
    sam_model_type: str = "vit_b"
    sam_points_per_side: int = 64
    sam_pred_iou_thresh: float = 0.8
    sam_stability_score_thresh: float = 0.85
    sam_crop_n_layers: int = 1
    sam_crop_n_points_downscale_factor: int = 2
    sam_min_mask_region_area: int = 70
    
    # Post-processing
    min_mask_area: int = 500
    clustering_eps: int = 20
    clustering_min_samples: int = 1
    
    # Paths
    sam_checkpoint_path: str = "models/sam_vit_b.pth"
    classifier_model_path: str = "models/InceptionV3_model.h5"
    gardiner_list_path: str = "data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx"
    
    # Output
    base_output_dir: str = "output"
    symbols_subdir: str = "extracted_symbols"
    save_intermediate: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/pipeline.log"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        
        if 'image_processing' in config_dict:
            img_config = config_dict['image_processing']
            flat_config['image_size'] = tuple(img_config.get('input_size', [512, 512]))
            flat_config['crop_size'] = tuple(img_config.get('crop_size', [299, 299]))
            flat_config['gaussian_kernel'] = tuple(img_config.get('gaussian_kernel', [5, 5]))
        
        if 'sam' in config_dict:
            sam_config = config_dict['sam']
            flat_config['sam_model_type'] = sam_config.get('model_type', 'vit_b')
            flat_config['sam_points_per_side'] = sam_config.get('points_per_side', 64)
            flat_config['sam_pred_iou_thresh'] = sam_config.get('pred_iou_thresh', 0.8)
            flat_config['sam_stability_score_thresh'] = sam_config.get('stability_score_thresh', 0.85)
            flat_config['sam_crop_n_layers'] = sam_config.get('crop_n_layers', 1)
            flat_config['sam_crop_n_points_downscale_factor'] = sam_config.get('crop_n_points_downscale_factor', 2)
            flat_config['sam_min_mask_region_area'] = sam_config.get('min_mask_region_area', 70)
            flat_config['sam_checkpoint_path'] = sam_config.get('checkpoint_path', 'models/sam_vit_b.pth')
        
        if 'post_processing' in config_dict:
            post_config = config_dict['post_processing']
            flat_config['min_mask_area'] = post_config.get('min_mask_area', 500)
            flat_config['clustering_eps'] = post_config.get('clustering_eps', 20)
            flat_config['clustering_min_samples'] = post_config.get('clustering_min_samples', 1)
        
        if 'paths' in config_dict:
            paths_config = config_dict['paths']
            flat_config['classifier_model_path'] = paths_config.get('classifier_model', 'models/InceptionV3_model.h5')
            flat_config['gardiner_list_path'] = paths_config.get('gardiner_list', 'data/Alan_Gardiners_List_of_Hieroglyphic_Signs.xlsx')
        
        if 'output' in config_dict:
            output_config = config_dict['output']
            flat_config['base_output_dir'] = output_config.get('base_directory', 'output')
            flat_config['symbols_subdir'] = output_config.get('symbols_subdirectory', 'extracted_symbols')
            flat_config['save_intermediate'] = output_config.get('save_intermediate_results', True)
        
        if 'logging' in config_dict:
            log_config = config_dict['logging']
            flat_config['log_level'] = log_config.get('level', 'INFO')
            flat_config['log_file'] = log_config.get('file', 'logs/pipeline.log')
        
        return cls(**flat_config)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'image_processing': {
                'input_size': list(self.image_size),
                'crop_size': list(self.crop_size),
                'gaussian_kernel': list(self.gaussian_kernel)
            },
            'sam': {
                'model_type': self.sam_model_type,
                'checkpoint_path': self.sam_checkpoint_path,
                'points_per_side': self.sam_points_per_side,
                'pred_iou_thresh': self.sam_pred_iou_thresh,
                'stability_score_thresh': self.sam_stability_score_thresh,
                'crop_n_layers': self.sam_crop_n_layers,
                'crop_n_points_downscale_factor': self.sam_crop_n_points_downscale_factor,
                'min_mask_region_area': self.sam_min_mask_region_area
            },
            'post_processing': {
                'min_mask_area': self.min_mask_area,
                'clustering_eps': self.clustering_eps,
                'clustering_min_samples': self.clustering_min_samples
            },
            'paths': {
                'classifier_model': self.classifier_model_path,
                'gardiner_list': self.gardiner_list_path
            },
            'output': {
                'base_directory': self.base_output_dir,
                'symbols_subdirectory': self.symbols_subdir,
                'save_intermediate_results': self.save_intermediate
            },
            'logging': {
                'level': self.log_level,
                'file': self.log_file
            }
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)