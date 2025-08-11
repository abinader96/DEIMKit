"""
Callback system for DEIMKit trainer.
Provides hooks for external integrations like MLflow.
"""

from abc import ABC
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TrainerCallback(ABC):
    """Base class for trainer callbacks"""
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training"""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, train_stats: Dict[str, float], eval_stats: Dict[str, Any], **kwargs):
        """Called at the end of each epoch"""
        pass
    
    def on_checkpoint_save(self, trainer, checkpoint_path: Path, metrics: Dict[str, Any], **kwargs):
        """Called when a checkpoint is saved"""
        pass


class MLflowCallback(TrainerCallback):
    """MLflow integration for comprehensive experiment tracking"""
    
    def __init__(self, experiment_name: str = "deim-training", run_name: Optional[str] = None, log_models: bool = True):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Please install it with: pip install mlflow"
            )
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_models = log_models
        self.run = None
        
    def on_train_begin(self, trainer, **kwargs):
        """Start MLflow run and log all hyperparameters"""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)
        
        # Log all configuration parameters
        config_dict = trainer.config.__dict__.copy()
        
        # Flatten nested configurations for MLflow
        flat_params = self._flatten_dict(config_dict)
        
        # Log hyperparameters (MLflow has a limit, so we'll log the most important ones)
        important_params = {
            k: v for k, v in flat_params.items() 
            if any(key in k for key in ['epochs', 'batch_size', 'lr', 'optimizer', 'model', 'image_size', 'num_classes', 'patience'])
            and isinstance(v, (int, float, str, bool))  # Only log simple types
        }
        
        # Ensure we don't exceed MLflow's parameter limit
        if len(important_params) > 100:
            # Sort by key and take first 100
            important_params = dict(sorted(important_params.items())[:100])
        
        mlflow.log_params(important_params)
        
        # Log the full config as an artifact
        config_path = Path(trainer.output_dir) / "config.yml"
        trainer.config.save(config_path)
        mlflow.log_artifact(str(config_path), "configs")
        
        # Log model architecture summary
        model_summary_path = Path(trainer.output_dir) / "model_architecture.txt"
        with open(model_summary_path, 'w') as f:
            f.write(str(trainer.model))
        mlflow.log_artifact(str(model_summary_path), "model_info")
        
        # Log number of parameters
        n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        mlflow.log_metric("model_parameters", n_params)
        
        print(f"MLflow run started: {self.run.info.run_id}")
    
    def on_epoch_end(self, trainer, epoch: int, train_stats: Dict[str, float], eval_stats: Dict[str, Any], **kwargs):
        """Log all training and evaluation metrics"""
        # Log training metrics
        for key, value in train_stats.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"train_{key}", value, step=epoch)
        
        # Log evaluation metrics
        if 'coco_eval_bbox' in eval_stats:
            bbox_stats = eval_stats['coco_eval_bbox']
            metric_names = [
                'mAP_50_95', 'mAP_50', 'mAP_75', 
                'mAP_small', 'mAP_medium', 'mAP_large',
                'mAR_1', 'mAR_10', 'mAR_100',
                'mAR_small', 'mAR_medium', 'mAR_large'
            ]
            
            for i, name in enumerate(metric_names):
                if i < len(bbox_stats):
                    mlflow.log_metric(f"val_{name}", bbox_stats[i], step=epoch)
            
            # Calculate and log F1 scores
            if len(bbox_stats) >= 9:
                # F1 = 2 * (precision * recall) / (precision + recall)
                if bbox_stats[0] + bbox_stats[8] > 0:
                    f1_50_95 = 2 * (bbox_stats[0] * bbox_stats[8]) / (bbox_stats[0] + bbox_stats[8])
                    mlflow.log_metric("val_F1_50_95", f1_50_95, step=epoch)
                
                if bbox_stats[1] + bbox_stats[8] > 0:
                    f1_50 = 2 * (bbox_stats[1] * bbox_stats[8]) / (bbox_stats[1] + bbox_stats[8])
                    mlflow.log_metric("val_F1_50", f1_50, step=epoch)
                
                if bbox_stats[2] + bbox_stats[8] > 0:
                    f1_75 = 2 * (bbox_stats[2] * bbox_stats[8]) / (bbox_stats[2] + bbox_stats[8])
                    mlflow.log_metric("val_F1_75", f1_75, step=epoch)
    
    def on_checkpoint_save(self, trainer, checkpoint_path: Path, metrics: Dict[str, Any], **kwargs):
        """Log model checkpoints as artifacts"""
        if self.log_models and checkpoint_path.exists():
            # Log the checkpoint
            mlflow.log_artifact(str(checkpoint_path), "checkpoints")
            
            # If it's the best model, also register it
            if "best" in str(checkpoint_path):
                try:
                    # Log the model with MLflow
                    mlflow.pytorch.log_model(
                        trainer.model,
                        "best_model",
                        registered_model_name=f"{self.experiment_name}_best_model"
                    )
                    print(f"Model registered in MLflow: {self.experiment_name}_best_model")
                except Exception as e:
                    # Just log the error, don't fail training
                    print(f"Warning: Could not register model with MLflow: {e}")
    
    def on_train_end(self, trainer, **kwargs):
        """Log final artifacts and close MLflow run"""
        # Log final checkpoint
        final_checkpoint = trainer.output_dir / "checkpoint_final.pth"
        if final_checkpoint.exists():
            mlflow.log_artifact(str(final_checkpoint), "checkpoints")
        
        # Log best checkpoint separately
        best_checkpoint = trainer.output_dir / "best.pth"
        if best_checkpoint.exists():
            mlflow.log_artifact(str(best_checkpoint), "checkpoints")
        
        # Log any tensorboard event files
        tb_files = list(trainer.output_dir.glob("events.out.tfevents.*"))
        for tb_file in tb_files:
            mlflow.log_artifact(str(tb_file), "tensorboard")
        
        # End the MLflow run
        mlflow.end_run()
        print(f"MLflow run ended: {self.run.info.run_id}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            elif v is not None and not callable(v):
                try:
                    # Try to convert to string for logging
                    items.append((new_key, str(v)))
                except:
                    # Skip if can't convert
                    pass
        return dict(items)
