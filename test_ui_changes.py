"""
Test script to verify the UI changes for Qwen Image LoRA GUI
"""
import sys
import os
import traceback

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules import correctly"""
    print("Testing imports...")
    try:
        # Import the main module
        from musubi_tuner_gui.qwen_image_lora_gui import (
            QwenImageDataset, 
            QwenImageModel,
            qwen_image_gui_actions,
            save_qwen_image_configuration,
            open_qwen_image_configuration
        )
        print("[OK] Successfully imported qwen_image_lora_gui module and classes")
        
        # Import command executor with notifications
        from musubi_tuner_gui.class_command_executor import CommandExecutor
        print("[OK] Successfully imported CommandExecutor class")
        
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_gradio_notifications():
    """Test that Gradio notification imports work"""
    print("\nTesting Gradio notification imports...")
    try:
        import gradio as gr
        
        # Test that notification functions exist
        assert hasattr(gr, 'Info'), "gr.Info not found"
        assert hasattr(gr, 'Warning'), "gr.Warning not found"
        
        print("[OK] Gradio notification functions are available")
        return True
    except AssertionError as e:
        print(f"[ERROR] Assertion error: {e}")
        return False
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_class_structure():
    """Test that the new class structure works"""
    print("\nTesting class structure...")
    try:
        import gradio as gr
        from musubi_tuner_gui.qwen_image_lora_gui import QwenImageDataset, QwenImageModel
        from musubi_tuner_gui.class_gui_config import GUIConfig
        
        # Create a dummy config
        config = GUIConfig()
        
        # Test class definitions exist and have the right methods
        print("  Checking QwenImageDataset class...")
        assert hasattr(QwenImageDataset, 'initialize_ui_components'), "initialize_ui_components method not found"
        assert hasattr(QwenImageDataset, 'setup_dataset_ui_events'), "setup_dataset_ui_events method not found"
        print("  [OK] QwenImageDataset class definition is correct")
        
        print("  Checking QwenImageModel class...")
        assert hasattr(QwenImageModel, 'initialize_ui_components'), "initialize_ui_components method not found"
        assert hasattr(QwenImageModel, 'setup_model_ui_events'), "setup_model_ui_events method not found"
        print("  [OK] QwenImageModel class definition is correct")
        
        # Test instantiation within a Gradio context
        print("  Testing instantiation within Gradio context...")
        with gr.Blocks() as demo:
            dataset = QwenImageDataset(headless=True, config=config)
            model = QwenImageModel(headless=True, config=config)
            
            # Check key attributes were created
            assert hasattr(dataset, 'dataset_config'), "dataset_config attribute not found"
            assert hasattr(model, 'dit'), "dit attribute not found"
            assert hasattr(model, 'vae'), "vae attribute not found"
            
        print("  [OK] Classes instantiate correctly within Gradio context")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error testing class structure: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing UI Changes for Qwen Image LoRA GUI")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_gradio_notifications,
        test_class_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    if all(results):
        print("[OK] All tests passed successfully!")
        print("\nSummary of changes:")
        print("1. Split Qwen Image Model Settings into two panels:")
        print("   - Qwen Image Training Dataset (top)")
        print("   - Qwen Image Model Settings (below)")
        print("2. Save/Load functionality maintained for both panels")
        print("3. Added Gradio notifications:")
        print("   - Info notification when training starts")
        print("   - Warning notification when training is cancelled")
        print("   - Info/Warning for caching operations")
        print("   - Success/Error notifications when training completes")
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)