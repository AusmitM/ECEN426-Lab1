# Differential Power Analysis (DPA) Attack on AES - Python Implementation

This directory contains Python implementations of Differential Power Analysis (DPA) attack on AES, converted from the original MATLAB code.

## Files

- `DPA_incomplete.py` - Student version with TODO sections to complete
- `aes_power_data.mat` - Power trace data containing plaintext, ciphertext, traces, and S-box

## Requirements

Install the required Python packages:
```bash
pip3 install --break-system-packages scipy numpy matplotlib
```

## Data Structure

The `aes_power_data.mat` file contains:
- `plain_text`: 200×16 array of plaintext bytes
- `cipher_text`: 200×16 array of corresponding ciphertext bytes  
- `traces`: 200×40000 array of power consumption traces
- `sbox`: 1×256 array containing the AES S-box values

## DPA Attack Overview

The DPA attack works by:
1. **Hypothesis Generation**: For each possible key byte (0-255), predict intermediate values
2. **Power Model**: Use a simple power model (LSB of S-box output) to predict power consumption
3. **Statistical Analysis**: Compute Difference of Means (DoM) between two trace groups
4. **Key Recovery**: The correct key byte produces the highest DoM peak

## Student Version (`DPA_incomplete.py`)

This version has several TODO sections for students to complete:

1. **S-box Lookup**: Complete the intermediate value calculation
   ```python
   y[i,:] = sbox[np.bitwise_xor(key_guess, input_plaintext[i])]
   ```

2. **Power Consumption Model**: Extract the least significant bit
   ```python
   power_consumption = np.bitwise_and(y, 1)
   ```

3. **Difference of Means**: Implement the core DPA calculation
   ```python
   group_0_indices = np.where(selec_col == 0)[0]
   group_1_indices = np.where(selec_col == 1)[0]
   mean_group_0 = np.mean(traces[group_0_indices, :], axis=0)
   mean_group_1 = np.mean(traces[group_1_indices, :], axis=0)
   DoM[col, :] = mean_group_1 - mean_group_0
   ```

### Expected Output

When run successfully, the complete version should output something like:
```
Full recovered key (hex):
00 2a 22 33 44 55 66 77 04 99 aa bb b5 91 cc ff
```

## Usage

For students:
```bash
python3 DPA_incomplete.py  # Complete the TODOs first
```# ECEN426-Lab1
