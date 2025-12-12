import sys
import os
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.trainer import Trainer
from pipeline.validator import Validator
from core import LogisticRegressionModel, RandomForestModel, SVMModel, DecisionTreeModel
from core.dataset import RespiratoryInfectionDataset
from core.base_model import BaseModel


# Configuration constants
DATASET_PATH = 'data/clclinical_respiratory_infection_dataset.csv'
MODELS_DIR = 'Models'
DISPLAY_WIDTH = 70
NUM_SAMPLE_PREDICTIONS = 5
RANDOM_STATE = 42


@dataclass
class ModelConfig:
    """Configuration for a model to train."""
    model_class: type
    model_name: str
    display_name: str
    params: Dict[str, Any]


# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        model_class=LogisticRegressionModel,
        model_name="logistic_regression",
        display_name="Logistic Regression",
        params={"max_iter": 1000, "random_state": RANDOM_STATE}
    ),
    ModelConfig(
        model_class=RandomForestModel,
        model_name="random_forest",
        display_name="Random Forest",
        params={"n_estimators": 100, "max_depth": 10, "random_state": RANDOM_STATE}
    ),
    ModelConfig(
        model_class=SVMModel,
        model_name="svm",
        display_name="Support Vector Machine",
        params={"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": RANDOM_STATE}
    ),
    ModelConfig(
        model_class=DecisionTreeModel,
        model_name="decision_tree",
        display_name="Decision Tree",
        params={"max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": RANDOM_STATE}
    )
]


def print_header(title: str, width: int = DISPLAY_WIDTH) -> None:
    """Print a formatted header."""
    if title:
        print("=" * width)
        print(title)
        print("=" * width)
    else:
        print("=" * width)


def print_separator(width: int = DISPLAY_WIDTH) -> None:
    """Print a separator line."""
    print("-" * width)


def load_dataset(csv_path: str = DATASET_PATH) -> Tuple[Any, Any, List[str]]:
    """Load and prepare the dataset.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    print(f"\n[1/4] Loading Dataset...")
    print_separator()
    
    dataset = RespiratoryInfectionDataset(csv_path=csv_path)
    dataset.load()
    dataset.prepare()
    X, y = dataset.get_data()
    feature_names = dataset.get_feature_names()
    
    print(f"\nDataset loaded with {len(X)} samples and {len(feature_names)} features")
    return X, y, feature_names


def train_models(X: Any, y: Any, models_dir: str = MODELS_DIR) -> Dict[str, Tuple[BaseModel, str]]:
    """Train all configured models.
    
    Returns:
        Dictionary mapping model_name to (model, model_path) tuple
    """
    print(f"\n[2/4] Training Models...")
    print_separator()
    
    trainer = Trainer(models_dir=models_dir)
    trained_models = {}
    
    for config in MODEL_CONFIGS:
        print(f"\n>>> Training {config.display_name} Model:")
        model = config.model_class(**config.params)
        trained_model, model_path = trainer.train(
            model,
            X, y,
            model_name=config.model_name,
            save_model=True
        )
        trained_models[config.model_name] = (trained_model, model_path)
    
    return trained_models


def validate_models(
    trained_models: Dict[str, Tuple[BaseModel, str]],
    X: Any,
    y: Any
) -> Dict[str, Tuple[Validator, Dict[str, Any]]]:
    """Validate all trained models.
    
    Returns:
        Dictionary mapping model_name to (validator, metrics) tuple
    """
    print(f"\n[3/4] Validating Models...")
    print_separator()
    
    validation_results = {}
    
    for config in MODEL_CONFIGS:
        model_name = config.model_name
        model, _ = trained_models[model_name]
        
        print(f"\n>>> Validating {config.display_name} Model:")
        validator = Validator()
        validator.load_model(model=model)
        metrics = validator.validate(X, y, verbose=True)
        
        validation_results[model_name] = (validator, metrics)
    
    return validation_results


def compare_models(validation_results: Dict[str, Tuple[Validator, Dict[str, Any]]]) -> str:
    """Compare models and display summary.
    
    Returns:
        Name of the best model
    """
    print("\n")
    print_header("Model Comparison Summary")
    
    # Get display names for each model
    model_display_names = {config.model_name: config.display_name for config in MODEL_CONFIGS}
    
    # Print header row
    header = f"{'Metric':<15}"
    for config in MODEL_CONFIGS:
        header += f" {config.display_name:<25}"
    print(f"\n{header}")
    print_separator()
    
    # Get all metrics
    metrics_data = {}
    for model_name, (_, metrics) in validation_results.items():
        metrics_data[model_name] = metrics
    
    # Print each metric
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric_name in metric_names:
        row = f"{metric_name.capitalize():<15}"
        for config in MODEL_CONFIGS:
            value = metrics_data[config.model_name][metric_name]
            row += f" {value:<25.4f}"
        print(row)
    
    # Determine best model
    best_model_name = max(
        validation_results.keys(),
        key=lambda name: validation_results[name][1]['accuracy']
    )
    best_display_name = model_display_names[best_model_name]
    
    print(f"\n>>> Best Model (by accuracy): {best_display_name}")
    return best_model_name


def run_predictions(
    best_model_name: str,
    validation_results: Dict[str, Tuple[Validator, Dict[str, Any]]],
    X: Any,
    y: Any,
    num_samples: int = NUM_SAMPLE_PREDICTIONS
) -> None:
    """Run sample predictions using the best model."""
    print(f"\n[4/4] Single Patient Prediction Examples")
    print_separator()
    
    best_validator, _ = validation_results[best_model_name]
    
    print("\nSample Predictions:")
    for i in range(min(num_samples, len(X))):
        patient_data = X[i]
        actual_label = y[i]
        
        result = best_validator.predict_single(patient_data, return_proba=True)
        
        print(f"\nPatient {i+1}:")
        print(f"  Actual: {'Infected' if actual_label == 1 else 'Not Infected'}")
        print(f"  Predicted: {'Infected' if result['prediction'] == 1 else 'Not Infected'}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Correct: {'✓' if result['prediction'] == actual_label else '✗'}")


def print_completion_summary(trained_models: Dict[str, Tuple[BaseModel, str]]) -> None:
    """Print completion summary with model paths."""
    print("\n")
    print_header("Pipeline Complete!")
    
    print(f"\nModels saved to:")
    for config in MODEL_CONFIGS:
        model_name = config.model_name
        _, model_path = trained_models[model_name]
        print(f"  - {model_path}")
    
    print("\nYou can load these models later using:")
    print("  validator = Validator()")
    print("  validator.load_model(model_path='Models/...')")


def main():
    """Main execution function."""
    print_header("Clinical Respiratory Infection Prediction System")
    
    # Load dataset
    X, y, feature_names = load_dataset()
    
    # Train models
    trained_models = train_models(X, y)
    
    # Validate models
    validation_results = validate_models(trained_models, X, y)
    
    # Compare models
    best_model_name = compare_models(validation_results)
    
    # Run predictions
    run_predictions(best_model_name, validation_results, X, y)
    
    # Print completion summary
    print_completion_summary(trained_models)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

