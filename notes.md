
1. **Why do we need "zero point"?**
   - The zero point acts as an offset or bias, shifting the scaled data to a proper position. It ensures that the real "0" is quantized without error.
   - Jacob's motivation for mentioning the zero point is that "Z is of the same type as quantized q." This means that when inferring in the quantized manner, the zero point (which has the same type as `q`) is padded (instead of the value "0") during zero-padding.
   - Essentially, the zero point ensures that the quantized zero aligns correctly with the real zero, maintaining consistency in the quantized representation¹.

2. **Why doesn't symmetric quantization need "zero point"?**
   - In symmetric quantization, both the floating range and the quantized range are symmetric. This symmetry allows us to set the zero point to 0.
   - For signed quantization, the floating-point values are quantized to signed fixed-point integers, with zero point = 0.
   - Similarly, for unsigned quantization, the floating-point values are quantized to unsigned fixed-point integers, still with zero point = 0.
   - The key distinction lies in the signs of the real, unquantized floating-point values, not the quantized ones. This approach ensures that the quantized ranges align correctly, even without an explicit zero point¹.

In summary, the zero point serves as an essential component in quantization, ensuring accurate representation and alignment between real and quantized values. Symmetric quantization leverages the inherent symmetry to avoid the need for an explicit zero point²³. 

Symmetric quantization schemes center the input range around 0, eliminating the need to calculate a zero-point offset. However, for skewed signals (like non-negative activations), this can result in suboptimal quantization resolution due to the clipping range including values that never appear in the input.

# Quantization 
## Different Granularities

Per-group quantization can require a lot more memory.

Let's say we want to quantize a tensor in 4-bit and we choose group-size = 32 symmetric mode (z=0), and we store the scales in FP16

It means that we actually quantizing the tensor in 4.5 bits since we have:
- 4 bit (each element is store in 4 bit)
- 16 / 32 bit (scale in 16 bits for every 32 elements)

# Inference for Linear Quantization
In a neural network, we can quantize the weights but also the activation. Depending on what we quantize, the storage and the computation are not the same!

| Storage | Computation |
|---|---|
| Quantized Weight + Activation(eg: W8A32) | floating point arithmetics (FP32, FP16, BF16, ..)|
|Quantized Weight + Quantized Activation (eg: W8A8) | Integer based arithmetics (INT8, INT4, ..)

Note: We need to dequantize the weights to perform the floating point computation!
Integer based arithmetics is not supported by all hardware.
