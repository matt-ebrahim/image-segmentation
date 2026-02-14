"""
Test script to verify deep supervision functionality
"""
import torch
import torch.nn as nn
from model.unet import AttentionUNet
import sys

def test_deep_supervision_outputs():
    """Test 1: Verify deep supervision outputs in training mode"""
    print("\n" + "="*70)
    print("TEST 1: Deep Supervision Output Structure")
    print("="*70)
    
    # Create model with deep supervision
    model = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=True)
    model.train()
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 256, 256)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Model in training mode: {model.training}")
    print(f"Deep supervision enabled: {model.deep_supervision}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Check if output is a tuple
    if isinstance(output, tuple):
        print(f"\n✓ PASS: Model returns tuple with {len(output)} outputs")
        
        # Expected: (main_output, out2, out3, out4)
        if len(output) == 4:
            print("✓ PASS: Correct number of outputs (4)")
        else:
            print(f"✗ FAIL: Expected 4 outputs, got {len(output)}")
            return False
            
        # Check shapes
        print("\nOutput shapes:")
        for i, out in enumerate(output):
            print(f"  Output {i}: {out.shape}")
            
            # All outputs should have same batch size and num_classes
            if out.shape[0] != batch_size:
                print(f"✗ FAIL: Output {i} has wrong batch size")
                return False
            if out.shape[1] != 2:  # num_classes
                print(f"✗ FAIL: Output {i} has wrong number of classes")
                return False
                
        print("✓ PASS: All outputs have correct batch size and num_classes")
        
        # Check spatial dimensions
        # Main output should be full resolution (256x256)
        if output[0].shape[2:] == torch.Size([256, 256]):
            print("✓ PASS: Main output has correct spatial dimensions (256x256)")
        else:
            print(f"✗ FAIL: Main output has wrong spatial dimensions: {output[0].shape[2:]}")
            return False
            
        # Auxiliary outputs should have smaller spatial dimensions
        expected_sizes = [(64, 64), (128, 128), (256, 256)]  # out4, out3, out2
        for i, expected_size in enumerate(expected_sizes):
            actual_size = output[i+1].shape[2:]
            # Allow some tolerance for size differences due to architecture
            if actual_size[0] >= expected_size[0] * 0.8 and actual_size[1] >= expected_size[1] * 0.8:
                print(f"✓ PASS: Auxiliary output {i+1} has reasonable spatial dimensions: {actual_size}")
            else:
                print(f"  WARNING: Auxiliary output {i+1} has unexpected spatial dimensions: {actual_size} (expected ~{expected_size})")
        
        return True
    else:
        print("✗ FAIL: Model should return tuple in training mode with deep supervision")
        return False

def test_deep_supervision_inference_mode():
    """Test 2: Verify single output in inference mode"""
    print("\n" + "="*70)
    print("TEST 2: Inference Mode Output")
    print("="*70)
    
    # Create model with deep supervision
    model = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=True)
    model.eval()  # Set to evaluation mode
    
    # Create dummy input
    input_tensor = torch.randn(2, 1, 256, 256)
    
    print(f"Model in training mode: {model.training}")
    print(f"Deep supervision enabled: {model.deep_supervision}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Check if output is a single tensor
    if isinstance(output, torch.Tensor):
        print(f"✓ PASS: Model returns single tensor in eval mode")
        print(f"  Output shape: {output.shape}")
        
        if output.shape == torch.Size([2, 2, 256, 256]):
            print("✓ PASS: Output has correct shape")
            return True
        else:
            print(f"✗ FAIL: Output has wrong shape, expected [2, 2, 256, 256]")
            return False
    else:
        print("✗ FAIL: Model should return single tensor in eval mode")
        return False

def test_deep_supervision_disabled():
    """Test 3: Verify behavior when deep supervision is disabled"""
    print("\n" + "="*70)
    print("TEST 3: Deep Supervision Disabled")
    print("="*70)
    
    # Create model without deep supervision
    model = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=False)
    model.train()
    
    # Create dummy input
    input_tensor = torch.randn(2, 1, 256, 256)
    
    print(f"Model in training mode: {model.training}")
    print(f"Deep supervision enabled: {model.deep_supervision}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Check if output is a single tensor even in training mode
    if isinstance(output, torch.Tensor):
        print(f"✓ PASS: Model returns single tensor when deep supervision is disabled")
        print(f"  Output shape: {output.shape}")
        
        if output.shape == torch.Size([2, 2, 256, 256]):
            print("✓ PASS: Output has correct shape")
            return True
        else:
            print(f"✗ FAIL: Output has wrong shape")
            return False
    else:
        print("✗ FAIL: Model should return single tensor when deep supervision is disabled")
        return False

def test_deep_supervision_loss_computation():
    """Test 4: Verify loss computation with deep supervision"""
    print("\n" + "="*70)
    print("TEST 4: Loss Computation with Deep Supervision")
    print("="*70)
    
    from train import CombinedLoss
    
    # Create model with deep supervision
    model = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=True)
    model.train()
    
    # Create dummy input and target
    input_tensor = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256))
    
    # Create loss function
    criterion = CombinedLoss()
    
    # Forward pass
    predictions = model(input_tensor)
    
    print(f"Predictions is tuple: {isinstance(predictions, tuple)}")
    
    if isinstance(predictions, tuple):
        print(f"Number of predictions: {len(predictions)}")
        
        # Compute main loss
        main_pred = predictions[0]
        loss, dice_loss, ce_loss = criterion(main_pred, target)
        print(f"\nMain loss: {loss.item():.4f}")
        print(f"  Dice loss component: {dice_loss.item():.4f}")
        print(f"  CE loss component: {ce_loss.item():.4f}")
        
        # Compute auxiliary losses
        aux_losses = []
        aux_weight = 0.25
        total_aux_loss = 0
        
        for i in range(1, len(predictions)):
            # Downsample target to match auxiliary output size
            aux_pred = predictions[i]
            aux_target = torch.nn.functional.interpolate(target.float(), size=aux_pred.shape[2:], mode='nearest')
            aux_loss, _, _ = criterion(aux_pred, aux_target)
            aux_losses.append(aux_loss.item())
            total_aux_loss += aux_weight * aux_loss.item()
            print(f"Auxiliary loss {i}: {aux_loss.item():.4f} (weighted: {aux_weight * aux_loss.item():.4f})")
        
        # Compute total loss
        total_loss = loss.item() + total_aux_loss
        print(f"\nTotal loss (main + weighted auxiliaries): {total_loss:.4f}")
        
        # Verify all losses are positive and finite
        all_losses_valid = True
        if not (torch.isfinite(loss) and loss.item() > 0):
            print("✗ FAIL: Main loss is not valid")
            all_losses_valid = False
            
        for i, aux_loss in enumerate(aux_losses):
            if not (aux_loss > 0 and torch.isfinite(torch.tensor(aux_loss))):
                print(f"✗ FAIL: Auxiliary loss {i+1} is not valid")
                all_losses_valid = False
        
        if all_losses_valid:
            print("✓ PASS: All losses are positive and finite")
            return True
        else:
            return False
    else:
        print("✗ FAIL: Expected tuple output for loss computation test")
        return False

def test_deep_supervision_gradients():
    """Test 5: Verify gradients flow through auxiliary outputs"""
    print("\n" + "="*70)
    print("TEST 5: Gradient Flow Through Auxiliary Outputs")
    print("="*70)
    
    from train import CombinedLoss
    
    # Create model with deep supervision
    model = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=True)
    model.train()
    
    # Create dummy input and target
    input_tensor = torch.randn(2, 1, 256, 256, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 256, 256))
    
    # Create loss function
    criterion = CombinedLoss()
    
    # Forward pass
    predictions = model(input_tensor)
    
    if isinstance(predictions, tuple):
        # Compute losses
        main_pred = predictions[0]
        loss, _, _ = criterion(main_pred, target)
        
        # Add auxiliary losses
        aux_weight = 0.25
        for i in range(1, len(predictions)):
            # Downsample target to match auxiliary output size
            aux_pred = predictions[i]
            aux_target = torch.nn.functional.interpolate(target.float(), size=aux_pred.shape[2:], mode='nearest')
            aux_loss, _, _ = criterion(aux_pred, aux_target)
            loss += aux_weight * aux_loss
        
        print(f"Total loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist in deep supervision layers
        has_gradients = True
        
        if hasattr(model, 'deep_supervision_out4'):
            if model.deep_supervision_out4.weight.grad is not None:
                grad_norm = model.deep_supervision_out4.weight.grad.norm().item()
                print(f"✓ deep_supervision_out4 gradient norm: {grad_norm:.6f}")
            else:
                print("✗ FAIL: No gradient for deep_supervision_out4")
                has_gradients = False
                
        if hasattr(model, 'deep_supervision_out3'):
            if model.deep_supervision_out3.weight.grad is not None:
                grad_norm = model.deep_supervision_out3.weight.grad.norm().item()
                print(f"✓ deep_supervision_out3 gradient norm: {grad_norm:.6f}")
            else:
                print("✗ FAIL: No gradient for deep_supervision_out3")
                has_gradients = False
                
        if hasattr(model, 'deep_supervision_out2'):
            if model.deep_supervision_out2.weight.grad is not None:
                grad_norm = model.deep_supervision_out2.weight.grad.norm().item()
                print(f"✓ deep_supervision_out2 gradient norm: {grad_norm:.6f}")
            else:
                print("✗ FAIL: No gradient for deep_supervision_out2")
                has_gradients = False
        
        if has_gradients:
            print("✓ PASS: Gradients flow through all auxiliary output layers")
            return True
        else:
            return False
    else:
        print("✗ FAIL: Expected tuple output")
        return False

def test_deep_supervision_layers_exist():
    """Test 6: Verify deep supervision layers are created correctly"""
    print("\n" + "="*70)
    print("TEST 6: Deep Supervision Layer Existence")
    print("="*70)
    
    # Test with deep supervision enabled
    model_with_ds = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=True)
    
    print("Model with deep supervision=True:")
    has_all_layers = True
    
    if hasattr(model_with_ds, 'deep_supervision_out4'):
        print(f"  ✓ deep_supervision_out4 exists: {model_with_ds.deep_supervision_out4}")
    else:
        print("  ✗ deep_supervision_out4 missing")
        has_all_layers = False
        
    if hasattr(model_with_ds, 'deep_supervision_out3'):
        print(f"  ✓ deep_supervision_out3 exists: {model_with_ds.deep_supervision_out3}")
    else:
        print("  ✗ deep_supervision_out3 missing")
        has_all_layers = False
        
    if hasattr(model_with_ds, 'deep_supervision_out2'):
        print(f"  ✓ deep_supervision_out2 exists: {model_with_ds.deep_supervision_out2}")
    else:
        print("  ✗ deep_supervision_out2 missing")
        has_all_layers = False
    
    # Test with deep supervision disabled
    model_without_ds = AttentionUNet(in_channels=1, num_classes=2, deep_supervision=False)
    
    print("\nModel with deep supervision=False:")
    layers_correctly_absent = True
    
    if not hasattr(model_without_ds, 'deep_supervision_out4'):
        print("  ✓ deep_supervision_out4 correctly not created")
    else:
        print("  ✗ deep_supervision_out4 should not exist")
        layers_correctly_absent = False
    
    if has_all_layers and layers_correctly_absent:
        print("\n✓ PASS: Deep supervision layers correctly managed")
        return True
    else:
        print("\n✗ FAIL: Deep supervision layers not correctly managed")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DEEP SUPERVISION TESTING SUITE")
    print("="*70)
    
    tests = [
        ("Output Structure", test_deep_supervision_outputs),
        ("Inference Mode", test_deep_supervision_inference_mode),
        ("Disabled Mode", test_deep_supervision_disabled),
        ("Loss Computation", test_deep_supervision_loss_computation),
        ("Gradient Flow", test_deep_supervision_gradients),
        ("Layer Existence", test_deep_supervision_layers_exist),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Deep supervision is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the failures above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
