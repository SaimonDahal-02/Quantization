import torch
from linear_quantization import (
    linear_q_with_scale_and_zero_point,
    linear_dequantization,
)
from helper import plot_quantization_errors


def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max / q_max


def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)

    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor=tensor, scale=scale, zero_point=0, dtype=torch.int8
    )
    # in symmetric quantization zero point is 0
    return quantized_tensor, scale


if __name__ == "__main__":
    test_tensor = torch.randn((4, 4))
    print("Original Tensor:", test_tensor)

    quantized_tensor, scale = linear_q_symmetric(test_tensor)
    print("Quantized Tensor:", quantized_tensor)

    dequantized_tensor = linear_dequantization(
        quantized_tensor=quantized_tensor, scale=scale, zero_point=0
    )
    print("Dequantized Tensor:", dequantized_tensor)

    # Plot Quantization Errors
    plot_name = "symmetric_mode.png"
    plot_quantization_errors(
        original_tensor=test_tensor,
        quantized_tensor=quantized_tensor,
        dequantized_tensor=dequantized_tensor,
        plot_path=f"png/{plot_name}",
    )

    # Overall Quantization Error (MSE)
    overall_error = (dequantized_tensor - test_tensor).square().mean()
    print("Overall Quantization Error:", overall_error)
