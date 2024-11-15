# /// script
# dependencies = ["numpy", "safetensors"]
# ///

import numpy as np
from safetensors.numpy import save_file


arrs = {
    "arr_bool": np.array([[True, False], [True, False]], dtype=bool),
    "arr_f2": np.array([[1, 2, 3], [-139.20182, 5, 6]], dtype="<f2"),
    "arr_f4": np.array([[1, 2, 3], [-139.20182, 5, 6]], dtype="<f4"),
    "arr_f8": np.array([[1, 2, 3], [-139.20182, 5, 6]], dtype="<f8"),
    "arr_i1": np.array([[1, 2, 3], [-4, 5, 6]], dtype="<i1"),
    "arr_i2": np.array([[1, 2, 3], [-167, 5, 6]], dtype="<i2"),
    "arr_i4": np.array([[1, 2, 3], [-1393645220, 5, 6]], dtype="<i4"),
    "arr_i8": np.array([[1, 2, 3], [-1678172369384593271, 5, 6]], dtype="<i8"),
    "arr_u1": np.array([[1, 2, 3], [4, 5, 6]], dtype="<u1"),
    "arr_u2": np.array([[1, 2, 3], [167, 5, 6]], dtype="<u2"),
    "arr_u4": np.array([[1, 2, 3], [1393645220, 5, 6]], dtype="<u4"),
    "arr_u8": np.array([[1, 2, 3], [16781723693845932719, 5, 6]], dtype="<u8"),
    "multidim": np.random.rand(3, 4, 5),
    "scalar": np.asarray(42),
    "simple": np.random.rand(10),
}

for name, arr in arrs.items():
    print(name, arr.dtype)
    print(arr)
save_file(tensor_dict=arrs, filename="arrs.safetensors", metadata={"hello": "world"})
