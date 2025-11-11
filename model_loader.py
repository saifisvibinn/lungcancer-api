"""
Robust model loader with compatibility fixes for scikit-learn version mismatches.
"""
import joblib
import pickle
import sys
import warnings

class SklearnCompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler that handles scikit-learn compatibility issues."""
    
    def find_class(self, module, name):
        # Handle EuclideanDistance compatibility issue
        if module == 'sklearn.metrics._dist_metrics' and name == 'EuclideanDistance':
            try:
                # Try to import and patch the module
                import sklearn.metrics._dist_metrics as dist_metrics
                
                # Check if EuclideanDistance exists
                if not hasattr(dist_metrics, 'EuclideanDistance'):
                    # Try to create it from available classes
                    if hasattr(dist_metrics, 'EuclideanDistance32'):
                        # Create a class that acts like EuclideanDistance
                        class EuclideanDistanceWrapper(dist_metrics.EuclideanDistance32):
                            pass
                        dist_metrics.EuclideanDistance = EuclideanDistanceWrapper
                    elif hasattr(dist_metrics, 'EuclideanDistance64'):
                        class EuclideanDistanceWrapper(dist_metrics.EuclideanDistance64):
                            pass
                        dist_metrics.EuclideanDistance = EuclideanDistanceWrapper
                    else:
                        # Last resort: try to find it in neighbors module
                        try:
                            from sklearn.neighbors._dist_metrics import EuclideanDistance as ED
                            dist_metrics.EuclideanDistance = ED
                        except:
                            # Create a minimal stub class
                            class EuclideanDistanceStub:
                                def __init__(self, *args, **kwargs):
                                    pass
                            dist_metrics.EuclideanDistance = EuclideanDistanceStub
                
                return getattr(dist_metrics, 'EuclideanDistance')
            except Exception as e:
                warnings.warn(f"Could not patch EuclideanDistance: {e}")
                # Fallback: return a stub class
                class EuclideanDistanceStub:
                    def __init__(self, *args, **kwargs):
                        pass
                return EuclideanDistanceStub
        
        # For all other classes, use default behavior
        return super().find_class(module, name)


def load_model_with_compatibility(model_path):
    """
    Load a joblib model with compatibility fixes.
    
    Args:
        model_path: Path to the .joblib model file
        
    Returns:
        Loaded model object
    """
    try:
        # First, try to patch the module before loading
        try:
            import sklearn.metrics._dist_metrics as dist_metrics
            if not hasattr(dist_metrics, 'EuclideanDistance'):
                if hasattr(dist_metrics, 'EuclideanDistance32'):
                    dist_metrics.EuclideanDistance = dist_metrics.EuclideanDistance32
                elif hasattr(dist_metrics, 'EuclideanDistance64'):
                    dist_metrics.EuclideanDistance = dist_metrics.EuclideanDistance64
        except:
            pass
        
        # Try standard loading first
        try:
            return joblib.load(model_path)
        except (AttributeError, ModuleNotFoundError) as e:
            if 'EuclideanDistance' in str(e):
                # Try with custom unpickler
                warnings.warn("Using compatibility mode to load model...")
                try:
                    # Use joblib's internal file handling but with custom unpickler
                    import joblib.numpy_pickle
                    
                    # Open the file
                    with open(model_path, 'rb') as f:
                        # Try to use joblib's format detection
                        unpickler = SklearnCompatibilityUnpickler(f)
                        try:
                            return unpickler.load()
                        except:
                            # If that doesn't work, try monkey-patching more aggressively
                            # Re-import after patching
                            import importlib
                            import sklearn.metrics._dist_metrics
                            importlib.reload(sklearn.metrics._dist_metrics)
                            
                            # Patch again after reload
                            dist_metrics = sklearn.metrics._dist_metrics
                            if not hasattr(dist_metrics, 'EuclideanDistance'):
                                if hasattr(dist_metrics, 'EuclideanDistance32'):
                                    # Create a proper alias
                                    dist_metrics.EuclideanDistance = type('EuclideanDistance', 
                                                                         (dist_metrics.EuclideanDistance32,), {})
                            
                            # Try loading again
                            return joblib.load(model_path)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model even with compatibility mode: {e2}")
            else:
                raise
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")


def load_sklearn_model_safe(model_path, scaler_path=None):
    """
    Safely load sklearn model and scaler with compatibility fixes.
    
    Args:
        model_path: Path to model .joblib file
        scaler_path: Path to scaler .joblib file (optional)
        
    Returns:
        Tuple of (model, scaler) or (model, None) if scaler_path not provided
    """
    model = load_model_with_compatibility(model_path)
    scaler = None
    
    if scaler_path:
        try:
            scaler = load_model_with_compatibility(scaler_path)
        except Exception as e:
            warnings.warn(f"Could not load scaler: {e}")
    
    return model, scaler

