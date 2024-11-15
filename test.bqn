⟨ExtractMetadata,GetArrayNames,GetArray,SerializeArrays⟩←•Import"safetensors.bqn"

# Read a safetensors file generated from Python
≠bytes←•file.MapBytes"arrs.safetensors"
{•Out𝕩⋄•Show bytes⊸GetArray𝕩⋄⟨⟩}¨∧GetArrayNames bytes

# Serialize BQN arrays to a safetensors file
a←3‿2⥊1‿0‿1‿1‿0‿0
b←3‿4‿5⥊↕60
c←2‿3⥊3.14×↕6
"test.safetensors"•file.Bytes "arr_a"‿"arr_b"‿"arr_c" SerializeArrays a‿b‿c

# Read the file that was created
≠bytes↩•file.MapBytes"test.safetensors"
names←∧GetArrayNames bytes
{•Out𝕩⋄•Show bytes⊸GetArray𝕩⋄⟨⟩}¨names
! a≡bytes GetArray "arr_a"
! b≡bytes GetArray "arr_b"
! c≡bytes GetArray "arr_c"
