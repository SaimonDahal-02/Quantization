import torch

from helper import plot_quantization_errors
from linear_quantization import linear_q_with_scale_and_zero_point, linear_dequantization
from symmetric_mode import get_q_scale_symmetric

def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    
    output_dim = r_tensor.shape[dim]
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale

if __name__ == "__main__":
    test_tensor=torch.tensor(
        [[191.6, -13.5, 728.6],
        [92.14, 295.5,  -184],
        [0,     684.6, 245.5]]
    )
    ### along the rows (dim = 0)
    quantized_tensor_0, scale_0 = linear_q_symmetric_per_channel(
        test_tensor, dim=0)
    
    dequantized_tensor_0 = linear_dequantization(
    quantized_tensor_0, scale_0, 0)

    ### along the columns (dim = 1)
    quantized_tensor_1, scale_1 = linear_q_symmetric_per_channel(
        test_tensor, dim=1)
    
    dequantized_tensor_1 = linear_dequantization(
    quantized_tensor_1, scale_1, 0)
    
    # Plot Quantization Errors
    plot_name = "per_row.png"
    plot_quantization_errors(
        original_tensor=test_tensor,
        quantized_tensor=quantized_tensor_0,
        dequantized_tensor=dequantized_tensor_0,
        plot_path=f"png/{plot_name}",
    )

    # Plot Quantization Errors
    plot_name = "per_column.png"
    plot_quantization_errors(
        original_tensor=test_tensor,
        quantized_tensor=quantized_tensor_1,
        dequantized_tensor=dequantized_tensor_1,
        plot_path=f"png/{plot_name}",
    )

    # Overall Quantization Error (MSE) per row
    overall_error = (dequantized_tensor_0 - test_tensor).square().mean()
    print("Overall Quantization Error Per Row:", overall_error)

    # Overall Quantization Error (MSE) per column
    overall_error = (dequantized_tensor_1 - test_tensor).square().mean()
    print("Overall Quantization Error Per Column:", overall_error)