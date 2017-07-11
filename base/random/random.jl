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

function __init__()
    try
        srand()
    catch ex
        Base.showerror_nostdio(ex,
            "WARNING: Error during initialization of module Random")
    end
end

include("RNGs.jl")
include("seeding.jl")
include("numbers.jl")
include("ranges.jl")
include("collections.jl")
include("arrays.jl")
include("normal.jl")
include("uuid.jl")
include("sequences.jl")

## rand & rand! docstrings

"""
    rand([rng=GLOBAL_RNG], [S], [dims...])

Pick a random element or array of random elements from the set of values specified by `S`;
`S` can be

* an indexable collection (for example `1:n` or `['x','y','z']`),
* an `Associative` or `AbstractSet` object,
* a string (considered as a collection of characters), or
* a type: the set of values to pick from is then equivalent to `typemin(S):typemax(S)` for
  integers (this is not applicable to [`BigInt`](@ref)), and to ``[0, 1)`` for floating
  point numbers;

`S` defaults to [`Float64`](@ref).

# Examples

```julia-repl
julia> rand(Int, 2)
2-element Array{Int64,1}:
 1339893410598768192
 1575814717733606317

julia> rand(MersenneTwister(0), Dict(1=>2, 3=>4))
1=>2
```

!!! note
    The complexity of `rand(rng, s::Union{Associative,AbstractSet})`
    is linear in the length of `s`, unless an optimized method with
    constant complexity is available, which is the case for `Dict`,
    `Set` and `IntSet`. For more than a few calls, use `rand(rng,
    collect(s))` instead, or either `rand(rng, Dict(s))` or `rand(rng,
    Set(s))` as appropriate.
"""
rand

"""
    rand!([rng=GLOBAL_RNG], A, [S=eltype(A)])

Populate the array `A` with random values. If `S` is specified
(`S` can be a type or a collection, cf. [`rand`](@ref) for details),
the values are picked randomly from `S`.
This is equivalent to `copy!(A, rand(rng, S, size(A)))`
but without allocating a new array.

# Example

```jldoctest
julia> rng = MersenneTwister(1234);

julia> rand!(rng, zeros(5))
5-element Array{Float64,1}:
 0.590845
 0.766797
 0.566237
 0.460085
 0.794026
```
"""
rand!

end # module
