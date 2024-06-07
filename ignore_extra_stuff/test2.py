import numpy as np
import pandas as pd

# Updated detect_peaks_and_dips function
def detect_peaks_and_dips(df, peak_type):
    highs = df['high'].values
    lows = df['low'].values
    center_index = peak_type // 2
    peaks = []
    dips = []
    
    for i in range(center_index, len(highs) - center_index):
        peak_window = highs[i - center_index:i + center_index + 1]
        dip_window = lows[i - center_index:i + center_index + 1]
        
        if all(peak_window[center_index] > peak_window[j] for j in range(len(peak_window)) if j != center_index):
            peaks.append(highs[i])
        
        if all(dip_window[center_index] < dip_window[j] for j in range(len(dip_window)) if j != center_index):
            dips.append(lows[i])
    
    return peaks, dips

# Sample data as DataFrame
data = {
    'high': [10, 12, 15, 14, 13, 17, 16, 19, 18, 17],
    'low': [8, 7, 6, 9, 8, 11, 10, 9, 12, 11]
}
df = pd.DataFrame(data)

# Parameters
peak_type = 5

# data = pd.DataFrame({
#         'high': [100, 200, 300, 400, 500, 600, 500, 400, 300, 200, 100],
#         'low': [50, 150, 250, 350, 450, 550, 450, 350, 250, 150, 50]
#     })

# df = pd.DataFrame(data)

# peak_type = 3

# Detect peaks and dips
peaks, dips = detect_peaks_and_dips(df, peak_type)
result = detect_peaks_and_dips(df, peak_type)

# Verifying the results
print("Peaks detected:")
for peak in peaks:
    print(f"Value: {peak}")

print("\nDips detected:")
for dip in dips:
    print(f"Value: {dip}")

print("\nResult:", result)
