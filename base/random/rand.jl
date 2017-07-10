# This file is a part of Julia. License is MIT: https://julialang.org/license

## Global RNG

const GLOBAL_RNG = MersenneTwister(0)
globalRNG() = GLOBAL_RNG

# rand: a non-specified RNG defaults to GLOBAL_RNG

"""
    rand([rng=GLOBAL_RNG], [S], [dims...])

Pick a random element or array of random elements from the set of values specified by `S`; `S` can be

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
@inline rand() = rand(GLOBAL_RNG, CloseOpen)
@inline rand(T::Type) = rand(GLOBAL_RNG, T)
rand(dims::Dims) = rand(GLOBAL_RNG, dims)
rand(dims::Integer...) = rand(convert(Dims, dims))
rand(T::Type, dims::Dims) = rand(GLOBAL_RNG, T, dims)
rand(T::Type, d1::Integer, dims::Integer...) = rand(T, tuple(Int(d1), convert(Dims, dims)...))
rand!(A::AbstractArray) = rand!(GLOBAL_RNG, A)

rand(r::AbstractArray) = rand(GLOBAL_RNG, r)

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
rand!(A::AbstractArray, r::AbstractArray) = rand!(GLOBAL_RNG, A, r)

rand(r::AbstractArray, dims::Dims) = rand(GLOBAL_RNG, r, dims)
rand(r::AbstractArray, dims::Integer...) = rand(GLOBAL_RNG, r, convert(Dims, dims))
