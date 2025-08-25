#Show how one weight is stored in different precisions:
import numpy as np
import sys

weight = 0.123456789  

# Different representations
w32 = np.float32(weight)
w16 = np.float16(weight)
w8  = np.int8(weight * 100)   # simple fake quantization to int8
w4  = np.int8(weight * 10)    # fake quantization to 4-bit scale

print("Float32:", w32, "Size:", sys.getsizeof(w32))
print("Float16:", w16, "Size:", sys.getsizeof(w16))
print("Int8 (quantized):", w8, "Size:", sys.getsizeof(w8))
print("4-bit (simulated):", w4, "Size:", sys.getsizeof(w4))
