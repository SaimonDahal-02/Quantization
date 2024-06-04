import torch
from helper import plot_quantization_errors
from per_channel import linear_q_symmetric_per_channel
from linear_quantization import linear_dequantization


def linear_q_symmetric_per_group(tensor, group_size, dtype=torch.int8):
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2

    tensor = tensor.view(-1, group_size)

    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)

    quantized_tensor = quantized_tensor.view(t_shape)

    return quantized_tensor, scale


def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)

    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)

    dequantized_tensor = dequantized_tensor.view(q_shape)

    return dequantized_tensor


if __name__ == "__main__":
    test_tensor = torch.tensor(
        [[191.6, -13.5, 728.6], [92.14, 295.5, -184], [0, 684.6, 245.5]]
    )

    group_size = 3

    quantized_tensor, scale = linear_q_symmetric_per_group(
        test_tensor, group_size=group_size
    )

    dequantized_tensor = linear_dequantization_per_group(
        quantized_tensor, scale, group_size=group_size
    )

    # Plot Quantization Errors
    plot_name = f"per_group_{group_size}.png"
    plot_quantization_errors(
        original_tensor=test_tensor,
        quantized_tensor=quantized_tensor,
        dequantized_tensor=dequantized_tensor,
        plot_path=f"png/{plot_name}",
    )
    # Overall Quantization Error (MSE) per row
    overall_error = (dequantized_tensor - test_tensor).square().mean()
    print("Overall Quantization Error Per Group:", overall_error)
