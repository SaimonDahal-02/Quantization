import torch
from symmetric_mode import linear_q_symmetric

def quantized_linear_W8A32_without_bias(input, q_w, s_w, z_w):
    assert input.dtype == torch.float32
    assert q_w.dtype == torch.int8

    dequantized_weight = q_w.to(torch.float32) * s_w + z_w
    output = torch.nn.functional.linear(input, dequantized_weight)
    
    return output

if __name__ == "__main__":
    input = torch.tensor([1, 2, 3], dtype=torch.float32)

    weight = torch.tensor([[-2,   -1.13, 0.42],
                       [-1.51, 0.25, 1.62],
                       [0.23,  1.35, 2.15]])
    q_w, s_w  = linear_q_symmetric(weight)

    output = quantized_linear_W8A32_without_bias(input,
                                             q_w,
                                             s_w,
                                             0)
    
    print(f"This is the W8A32 output: {output}")

    fp32_output = torch.nn.functional.linear(input, weight)

    print(f"This is the output if we don't quantize: {fp32_output}")