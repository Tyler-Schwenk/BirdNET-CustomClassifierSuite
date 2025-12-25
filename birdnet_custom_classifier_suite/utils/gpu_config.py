"""GPU configuration utilities to prevent OOM errors on GPUs with limited memory."""

def configure_gpu_memory():
    """
    Enable memory growth for all GPUs to prevent TensorFlow from allocating
    all GPU memory at once.
    
    This is especially important for GPUs with limited VRAM (e.g., 4GB).
    Without this, TensorFlow may fail with OOM errors even when the model
    would fit with proper memory management.
    
    Call this BEFORE any TensorFlow operations (model loading, training, etc.).
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
                
                # Print GPU info
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                    
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"WARNING: GPU configuration warning: {e}")
        else:
            print("INFO: No GPU detected - running on CPU")
            
    except ImportError:
        print("WARNING: TensorFlow not available - GPU configuration skipped")
    except Exception as e:
        print(f"WARNING: GPU configuration error: {e}")


if __name__ == "__main__":
    # Test GPU configuration
    configure_gpu_memory()
    
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    except ImportError:
        print("TensorFlow not installed")
