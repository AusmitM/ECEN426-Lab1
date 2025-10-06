#!/usr/bin/env python3
"""
Differential Power Analysis (DPA) Attack on AES - Student Version
This is an incomplete version for students to complete.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load plaintext, ciphertext, traces, and sbox
data = scipy.io.loadmat('aes_power_data.mat')
plain_text = data['plain_text']
cipher_text = data['cipher_text']
traces = data['traces']
sbox = data['sbox'].flatten()  # Convert from (1,256) to (256,)

print(f"Loaded data:")
print(f"Plain text shape: {plain_text.shape}")
print(f"Cipher text shape: {cipher_text.shape}")
print(f"Traces shape: {traces.shape}")
print(f"S-box shape: {sbox.shape}")

bytes_recovered = np.zeros(16, dtype=np.uint8)
n_traces = 200

traces = traces[:n_traces, :]

## Launch DPA and compute DoM, size of DoM 256 x 40000
# Part 1

for i in range(16): 
    byte_to_attack = i  # Python uses 0-based indexing
    key_guess = np.arange(256, dtype=np.uint8)  # 0 to 255

    input_plaintext = plain_text[:, byte_to_attack]

    # TODO: Students need to complete this section
    # Hint: Create a matrix y where y[i,j] represents the S-box output
    # for trace i and key guess j
    y = np.zeros((200, 256), dtype=np.uint8)
    for i in range(200):
        # TODO: Fill in the S-box lookup
        # y[i,:] = sbox[???]
        y[i, :] = sbox[np.bitwise_xor(key_guess, input_plaintext[i])]
        pass

    # TODO: Students need to implement power consumption prediction
    # Hint: Extract the least significant bit from y
    power_consumption = np.zeros((200, 256), dtype=np.uint8)
    # power_consumption = ???
    power_consumption = np.bitwise_and(y, 1)

    # Part 2 - DPA attack
    DoM = np.zeros((256, traces.shape[1]))  # Difference of Means matrix

    maxDoM = 0
    maxNum = 0
    for col in range(256):
        # TODO: Students need to implement the difference of means calculation
        # Hint: Separate traces into two groups based on power_consumption
        # Group 0: where power_consumption[:, col] == 0
        # Group 1: where power_consumption[:, col] == 1
        # DoM[col, :] = mean(group1) - mean(group0)
        
        selec_col = power_consumption[:, col]
        group0 = np.where(selec_col == 0)[0]
        group1 = np.where(selec_col ==1)[0]
        mean0 = np.mean(traces[group0, :], axis=0)
        mean1 = np.mean(traces[group1, :], axis=0)
        DoM[col, :] = mean1-mean0
        if maxDoM<abs(mean1-mean0):
            maxDoM = abs(mean1-mean0)
            maxNum = col
        # TODO: Complete the DoM calculation
        pass

    

print("Bytes recovered (hex):")
print(' '.join([f'{b:02x}' for b in bytes_recovered]))

# Compare with the golden key (if known)
print("\nNote: Complete the implementation to recover the key bytes")

## Sample code for plots
OFFSET = 192  # for N=64, use 0, 64, 128, 192
N = 8  # for an NxN plot

fig, axes = plt.subplots(N, N, figsize=(12, 12))
fig.suptitle(f'DoM Plot (Offset: {OFFSET})')

for i in range(N):
    for j in range(N):
        key_candidate = (i * N + j + OFFSET) % 256
        if key_candidate < DoM.shape[0]:
            axes[i, j].plot(DoM[key_candidate, :])
            axes[i, j].set_title(f'Key: {key_candidate:02x}')
            axes[i, j].set_xlim(0, DoM.shape[1])

plt.tight_layout()
plt.savefig('dpa_analysis_incomplete.png', dpi=150, bbox_inches='tight')
plt.show()
