import sys
import torch
from helper import plot_quantization_errors


def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    # r = s(q-z)
    # q = int(round((r / s) + z))
    scale_and_shift_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scale_and_shift_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor


def linear_dequantization(quantized_tensor, scale, zero_point):
    dequantized_tensor = scale * (
        quantized_tensor.float() + zero_point
    )  # without float causes overflows and underflows
    return dequantized_tensor


def get_scale_and_zero_point(tensor, dtype=torch.int8):
    
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point


def linear_quantization(tensor, dtype=torch.int8):
    scale, zero_point = get_scale_and_zero_point(tensor, dtype=dtype)

    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor, scale, zero_point, dtype=dtype
    )

    return quantized_tensor, scale, zero_point


if __name__ == "__main__":
    ### a dummy tensor to test the implementation
    test_tensor = torch.randn((4,4))
    print("Original Tensor:", test_tensor)

    if len(sys.argv) > 1 and sys.argv[1] == "--random":
        print("Testing with random scale and zero point:")
        scale = 3.45
        zero_point = -3
    else:
        print("Testing with generated scale and zero point:")
        scale, zero_point = get_scale_and_zero_point(test_tensor)

    quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)
    print("Quantized Tensor:", quantized_tensor)

    # Dequantize the tensor
    dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)
    print("Dequantized Tensor:", dequantized_tensor)

    # Plot Quantization Errors
    plot_name = "random_scale_zero.png" if len(sys.argv) > 1 and sys.argv[1] == "--random" else "generated_scale_zero.png"
    plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor, plot_path=f"png/{plot_name}")

    # Overall Quantization Error (MSE)
    overall_error = (dequantized_tensor - test_tensor).square().mean()
    print("Overall Quantization Error:", overall_error)
