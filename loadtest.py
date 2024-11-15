# /// script
# dependencies = ["numpy", "safetensors"]
# ///

from safetensors.numpy import load_file

arrs = load_file("test.safetensors")
for name, arr in arrs.items():
    print(name, arr.dtype)
    print(arr)
