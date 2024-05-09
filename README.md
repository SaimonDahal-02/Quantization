---
## Linear Quantization

This script demonstrates linear quantization and dequantization of a given tensor using scale and zero-point parameters.

### Requirements
- Dependencies can be installed using `requirements.txt`. Run:
  ```bash
  pip install -r requirements.txt
  ```

### Usage
1. Clone the repository.
2. Ensure you have Python and PyTorch installed.
3. Install additional dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
4. Navigate to the directory containing `linear_quantization.py`.
5. Run the script with the following command:
   ```bash
   python linear_quantization.py [--random]
   ```
   - Use the `--random` flag to test with random scale and zero-point values.
   - Omit the flag to test with generated scale and zero-point values based on the input tensor.
6. View the printed results including the original tensor, quantized tensor, dequantized tensor, and overall quantization error.
7. Optionally, check the generated PNG file in the `png/` directory, showing the quantization errors.

### Example
```bash
python linear_quantization.py
```

### Notes
- The script `helper.py` contains auxiliary functions used for plotting quantization errors.

---