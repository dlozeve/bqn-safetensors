# bqn-safetensors

[BQN](https://mlochbaum.github.io/BQN/) library to read and write arrays stored in the [safetensors](https://github.com/huggingface/safetensors) format.

## Why safetensors?

Safetensors is a format to store multidimensional arrays (tensors) in
a safe, efficient way. Compared the NPY format (see
[bqn-npy](https://github.com/dlozeve/bqn-npy)), it has the advantage
of being able to store multiple named arrays in a single file, and
being able to read an array without loading the entire file in memory.

## Supported array types

This library can load arrays stored with any dtype supported by the
official library. However, it can only store arrays in 32-bit signed
integers (I32) and double-precision floating point (F64), which are
the native types of CBQN.

| Format   | Description                                                                          | Deserialization | Serialization |
|----------|--------------------------------------------------------------------------------------|-----------------|---------------|
| BOOL     | Boolean type                                                                         | ✅              | ❌            |
| U8       | Unsigned byte                                                                        | ✅              | ❌            |
| I8       | Signed byte                                                                          | ✅              | ❌            |
| F8\_E5M4 | [FP8](https://arxiv.org/abs/2209.05433)                                              | ✅              | ❌            |
| F8\_E4M3 | [FP8](https://arxiv.org/abs/2209.05433)                                              | ✅              | ❌            |
| I16      | Signed integer (16-bit)                                                              | ✅              | ❌            |
| U16      | Unsigned integer (16-bit)                                                            | ✅              | ❌            |
| F16      | Half-precision floating point                                                        | ✅              | ❌            |
| BF16     | [Brain floating point](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) | ✅              | ❌            |
| I32      | Signed integer (32-bit)                                                              | ✅              | ✅            |
| U32      | Unsigned integer (32-bit)                                                            | ✅              | ❌            |
| F32      | Floating point (32-bit)                                                              | ✅              | ❌            |
| F64      | Floating point (64-bit)                                                              | ✅              | ✅            |
| I64      | Signed integer (64-bit)                                                              | ✅              | ❌            |
| U64      | Unsigned integer (64-bit)                                                            | ✅              | ❌            |

## Usage

```bqn
⟨ExtractMetadata,GetArrayNames,GetArray,SerializeArrays⟩←•Import"safetensors.bqn"

# Use •file.MapBytes to avoid loading the entire file in memory, using memory-mapping instead.
bytes←•file.MapBytes"input.safetensors"

# Extract the metadata (a string -> string map) from the file.
ExtractMetadata bytes

# List the arrays present in the file.
GetArrayNames bytes

# Read an array from the file.
bytes GetArray "my_array"

# Create arrays and save them to a safetensors file.
a←3‿4‿5⥊↕60    # will be stored as I32
b←2‿3⥊3.14×↕6  # will be stored as F64
"output.safetensors"•file.Bytes "arr_a"‿"arr_b" SerializeArrays a‿b
```

## How to test

The script [`gentest.py`](gentest.py) generates a test safetensors
file (`arrs.safetensors`) containing arrays of various shapes and
dtypes. It requires `numpy` and `safetensors` dependencies. The
easiest way to run it is via [`uv run`](https://docs.astral.sh/uv/guides/scripts/),
which takes care of installing the dependencies in an isolated
environment automatically:

```
uv run gentest.py
```

The script [`test.bqn`](test.bqn) then reads this test file and
displays its contents to check that they are identical to the ones
reported by Python. It also creates another set of arrays, serializes
them to a `test.safetensors` file, and deserializes it again.

```
bqn test.bqn
```

Finally, the script [`loadtest.py`](loadtest.py) reads the
BQN-generated safetensors file to ensure that it is readable from
Python.

```
uv run loadtest.py
```
