"""
Comprehensive Test Suite for Fraud Detection Application
Tests models, data processing, and application functionality
"""

import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestModelValidation:
    """Test suite for validating ML models"""
    
    def test_models_exist(self):
        """Test that all required model files exist"""
        required_models = [
            'best_fraud_detection_model.pkl',
            'scaler.pkl'
        ]
        
        for model_file in required_models:
            assert Path(model_file).exists(), f"Required model {model_file} not found"
    
    def test_models_loadable(self):
        """Test that all models can be loaded without errors"""
        models = [
            'best_fraud_detection_model.pkl',
            'model_LightGBM.pkl',
            'model_Random_Forest_Baseline.pkl',
            'model_Random_Forest_Optimisé.pkl',
            'model_XGBoost.pkl',
            'scaler.pkl'
        ]
        
        for model_file in models:
            if Path(model_file).exists():
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    assert model is not None, f"Model {model_file} loaded but is None"
                except Exception as e:
                    pytest.fail(f"Failed to load {model_file}: {str(e)}")
    
    def test_model_prediction(self):
        """Test that the best model can make predictions"""
        with open('best_fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Create dummy test data (30 features as per credit card fraud detection)
        n_features = 30  # Typical for credit card fraud datasets
        test_data = np.random.randn(5, n_features)
        
        try:
            # Scale data
            scaled_data = scaler.transform(test_data)
            
            # Make predictions
            predictions = model.predict(scaled_data)
            
            assert predictions is not None
            assert len(predictions) == 5
            assert all(p in [0, 1] for p in predictions), "Predictions should be binary (0 or 1)"
            
        except Exception as e:
            pytest.fail(f"Model prediction failed: {str(e)}")
    
    def test_model_prediction_proba(self):
        """Test that the model can output probabilities"""
        with open('best_fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        n_features = 30
        test_data = np.random.randn(5, n_features)
        
        try:
            scaled_data = scaler.transform(test_data)
            probabilities = model.predict_proba(scaled_data)
            
            assert probabilities is not None
            assert probabilities.shape == (5, 2), "Should return probabilities for 2 classes"
            assert all(0 <= p <= 1 for row in probabilities for p in row), \
                "All probabilities should be between 0 and 1"
            
        except Exception as e:
            pytest.fail(f"Probability prediction failed: {str(e)}")


class TestDataProcessing:
    """Test suite for data processing functions"""
    
    def test_scaler_transform(self):
        """Test that scaler properly transforms data"""
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Test with sample data
        test_data = np.random.randn(10, 30)
        
        try:
            scaled_data = scaler.transform(test_data)
            
            assert scaled_data.shape == test_data.shape
            assert not np.isnan(scaled_data).any(), "Scaled data contains NaN values"
            assert not np.isinf(scaled_data).any(), "Scaled data contains infinite values"
            
        except Exception as e:
            pytest.fail(f"Scaler transformation failed: {str(e)}")


class TestApplicationRequirements:
    """Test suite for application dependencies"""
    
    def test_import_streamlit(self):
        """Test that Streamlit can be imported"""
        try:
            import streamlit
            assert streamlit is not None
        except ImportError:
            pytest.fail("Streamlit package not available")
    
    def test_import_pandas(self):
        """Test that Pandas can be imported"""
        try:
            import pandas
            assert pandas is not None
        except ImportError:
            pytest.fail("Pandas package not available")
    
    def test_import_sklearn(self):
        """Test that scikit-learn can be imported"""
        try:
            import sklearn
            assert sklearn is not None
        except ImportError:
            pytest.fail("Scikit-learn package not available")
    
    def test_import_plotly(self):
        """Test that Plotly can be imported"""
        try:
            import plotly
            assert plotly is not None
        except ImportError:
            pytest.fail("Plotly package not available")


class TestModelPerformance:
    """Test suite for model performance metrics"""
    
    def test_model_file_size(self):
        """Test that model files are within reasonable size limits"""
        max_size_mb = 500  # Maximum 500MB per model
        
        model_files = [
            'best_fraud_detection_model.pkl',
            'model_LightGBM.pkl',
            'model_Random_Forest_Baseline.pkl',
            'model_Random_Forest_Optimisé.pkl',
            'model_XGBoost.pkl',
            'scaler.pkl'
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                size_mb = Path(model_file).stat().st_size / (1024 * 1024)
                assert size_mb < max_size_mb, \
                    f"Model {model_file} is too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"


def test_visualization_files_exist():
    """Test that visualization files exist"""
    viz_files = [
        'confusion_matrix_LightGBM.png',
        'confusion_matrix_XGBoost.png',
        'model_comparison_bars.png',
        'model_comparison_radar.png'
    ]
    
    # These are optional, so we just check and warn
    for viz_file in viz_files:
        if not Path(viz_file).exists():
            print(f"Warning: Visualization file {viz_file} not found")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])