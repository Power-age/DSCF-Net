"""
Model profiling utilities: parameter counting and FLOPs calculation.
"""

import torch


def count_parameters(model, trainable_only=True):
    """Count model parameters in millions."""
    if trainable_only:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params / 1e6


def count_flops(model, input_shape=(1, 3, 256, 256)):
    """Count FLOPs using thop library.

    Args:
        model: PyTorch model.
        input_shape: input tensor shape.

    Returns:
        FLOPs in Giga (G).
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(*input_shape).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0.0


def profile_model(model, input_shape=(1, 3, 256, 256), device="cuda"):
    """Print model profiling summary.

    Args:
        model: PyTorch model.
        input_shape: input tensor shape.
        device: device for computation.

    Returns:
        (params_M, flops_G)
    """
    model = model.to(device)
    model.eval()

    params_m = count_parameters(model)
    flops_g = count_flops(model, input_shape)

    print(f"{'='*60}")
    print(f"Model Profile")
    print(f"{'='*60}")
    print(f"  Parameters : {params_m:.2f} M")
    print(f"  FLOPs      : {flops_g:.4f} G (@ {input_shape})")
    print(f"{'='*60}")

    return params_m, flops_g
