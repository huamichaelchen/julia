# This file is a part of Julia. License is MIT: https://julialang.org/license

module Random

using Base.dSFMT
using Base.GMP: Limb, MPZ
import Base: copymutable, copy, copy!, ==

export srand,
       rand, rand!,
       randn, randn!,
       randexp, randexp!,
       bitrand,
       randstring,
       randsubseq,randsubseq!,
       shuffle,shuffle!,
       randperm, randcycle,
       AbstractRNG, MersenneTwister, RandomDevice,
       GLOBAL_RNG, randjump


abstract type AbstractRNG end

abstract type FloatInterval end
mutable struct CloseOpen <: FloatInterval end
mutable struct Close1Open2 <: FloatInterval end

## initialization

function __init__()
    try
        srand()
    catch ex
        Base.showerror_nostdio(ex,
            "WARNING: Error during initialization of module Random")
    end
end

include("randomdevice.jl")
include("mersennetwister.jl")
include("srand.jl")
include("rand.jl")
include("numbers.jl")
include("ranges.jl")
include("collections.jl")
include("arrays.jl")
include("normal.jl")
include("uuid.jl")
include("sequences.jl")

end # module
