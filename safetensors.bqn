⟨ExtractMetadata,GetArrayNames,GetArray,SerializeArrays⟩⇐

⟨Parse,Export⟩←•Import"json.bqn"

ExtractHeader←{𝕊bytes:
  n←2⊸×⊸+˜´⟨8‿'c',1‿'u'⟩•bit._cast 8↑bytes
  ⟨Parse n↑8↓bytes,n+8⟩
}

JsonGet←{(⊑(⊏𝕨)⊐<𝕩)⊑1⊏𝕨}

ExtractMetadata←{𝕊bytes:
  header‿·←ExtractHeader bytes
  header JsonGet⎊⟨⟩ "__metadata__"
}

GetArrayNames←{𝕊bytes:
  header‿·←ExtractHeader bytes
  "__metadata__"⊸≢¨⊸/⊏header
}

# Valid for sizes 8, 16, and 32 (passed as 𝕨)
ParseUint←{2⊸×⊸+˜´˘⟨8‿'c',1‿'u'⟩•bit._cast˘ ∘‿(𝕨÷8)⥊𝕩}
ParseInt←{(-2⋆𝕨-1)+(2⋆𝕨)|(2⋆𝕨-1)+𝕨ParseUint𝕩}

ParseUint64←{(2⋆32)⊸×⊸+˜´˘ ∘‿2⥊32 ParseUInt 𝕩}
ParseInt64←{(2⋆32)⊸×⊸+˜´˘ ∘‿2⥊32 ParseInt 𝕩}

# Parse a floating point number
# e is the size of the exponent part
ParseFloat←{e𝕊bytes:
  n←⌽⟨8‿'c',1‿'u'⟩•bit._cast bytes
  s←(≠n)-e+1
  sign←1+2×-⊑n
  exponent←2⊸×⊸+˜´⌽e↑1↓n
  significand←2⊸×⊸+˜´⌽1∾e↓1↓n
  sign×(2⋆exponent-((2⋆e-1)-1))×significand÷2⋆s
}

dtypes←⟨
  "BOOL",    # Boolean type
  "U8",      # Unsigned byte
  "I8",      # Signed byte
  "F8_E5M2", # FP8 <https://arxiv.org/pdf/2209.05433.pdf>
  "F8_E4M3", # FP8 <https://arxiv.org/pdf/2209.05433.pdf>
  "I16",     # Signed integer (16-bit)
  "U16",     # Unsigned integer (16-bit)
  "F16",     # Half-precision floating point
  "BF16",    # Brain floating point
  "I32",     # Signed integer (32-bit)
  "U32",     # Unsigned integer (32-bit)
  "F32",     # Floating point (32-bit)
  "F64",     # Floating point (64-bit)
  "I64",     # Signed integer (64-bit)
  "U64",     # Unsigned integer (64-bit)
⟩
typeConversions←⟨
  ⟨8‿'c', 1‿'u'⟩•bit._cast, # BOOL
  8⊸ParseUint,              # U8
  ⟨8‿'c', 8‿'i'⟩•bit._cast, # I8
  5⊸ParseFloat˘∘‿1⊸⥊,       # F8_E5M2
  4⊸ParseFloat˘∘‿1⊸⥊,       # F8_E4M4
  ⟨8‿'c',16‿'i'⟩•bit._cast, # I16
  16⊸ParseUint,             # U16
  5⊸ParseFloat˘∘‿2⊸⥊,       # F16
  8⊸ParseFloat˘∘‿2⊸⥊,       # BF16
  ⟨8‿'c',32‿'i'⟩•bit._cast, # I32
  32⊸ParseUint,             # U32
  8⊸ParseFloat˘∘‿4⊸⥊,       # F32
  ⟨8‿'c',64‿'f'⟩•bit._cast, # F64
  ParseInt64,               # I64
  ParseUint64,              # U64
⟩

GetArray←{bytes𝕊name:
  header‿n←ExtractHeader bytes
  byteBuf←n↓bytes
  info←header JsonGet name
  s‿e←info JsonGet "data_offsets"
  shape←info JsonGet "shape"
  dtypeIdx←⊑dtypes⊐<info JsonGet "dtype"
  conv←dtypeIdx⊑typeConversions
  shape⥊Conv s↓e↑byteBuf
}

SerializeArray←{
  dtype←(∧´⌊⊸=⥊𝕩)⊑"F64"‿"I32"
  shape←≢𝕩
  data←(∧´⌊⊸=)◶⟨⟨64‿'f',8‿'c'⟩•bit._cast,⟨32‿'i',8‿'c'⟩•bit._cast⟩⥊𝕩
  dtype‿shape‿data
}

SerializeArrays←{names𝕊arrs:
  dtypes‿shapes‿datas←<˘⍉>SerializeArray¨arrs
  dataOffsets←<˘2↕0∾+`≠¨datas
  blocks←{𝕊name‿dtype‿shape‿dataOffset:
    ["dtype"‿"shape"‿"data_offsets",dtype‿shape‿dataOffset]
  }¨<˘⍉>names‿dtypes‿shapes‿dataOffsets
  header←[names,blocks]
  n←≠headerJson←Export header
  nEncoded←⟨32‿'i',8‿'c'⟩•bit._cast⟨n,0⟩
  nEncoded∾headerJson∾∾datas
}
