"""
Script to inspect TabPFN architecture and find attention layers.
Run after installing tabpfn: pip install tabpfn
"""

try:
    from tabpfn import TabPFNClassifier
    import torch
    
    print("Initializing TabPFN...")
    model = TabPFNClassifier(device='cpu')
    
    print(f"\nModel type: {type(model)}")
    print(f"\nModel class: {model.__class__.__name__}")
    
    # Check for common attributes
    attrs_to_check = [
        'model', 'model_', 'c', 'transformer', 'encoder',
        'transformer_encoder', 'predict_proba', 'fit'
    ]
    
    print("\nChecking attributes:")
    for attr in attrs_to_check:
        has_attr = hasattr(model, attr)
        print(f"  {attr}: {has_attr}")
        if has_attr:
            obj = getattr(model, attr)
            print(f"    Type: {type(obj)}")
    
    # Try to access internal model
    print("\nTrying to access internal model...")
    if hasattr(model, 'model'):
        inner = model.model
        print(f"  model.model type: {type(inner)}")
        if hasattr(inner, '__dict__'):
            print(f"  model.model attributes: {list(inner.__dict__.keys())[:10]}")
    
    if hasattr(model, 'c'):
        inner = model.c
        print(f"  model.c type: {type(inner)}")
        if isinstance(inner, torch.nn.Module):
            print("  model.c is a torch Module!")
            for name, module in inner.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    print(f"    Found attention layer: {name} ({type(module)})")
    
    print("\nPublic methods:")
    public_methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    print(f"  {public_methods[:15]}")
    
except ImportError as e:
    print(f"TabPFN not installed: {e}")
    print("Install with: pip install tabpfn")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
