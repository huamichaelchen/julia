# This file is a part of Julia. License is MIT: https://julialang.org/license

## random values from AbstractArray

rand(rng::AbstractRNG, r::AbstractArray) = @inbounds return r[rand(rng, 1:length(r))]
rand(r::AbstractArray) = rand(GLOBAL_RNG, r)

### arrays

function rand!(rng::AbstractRNG, A::AbstractArray, r::AbstractArray)
    g = RangeGenerator(1:(length(r)))
    for i in eachindex(A)
        @inbounds A[i] = r[rand(rng, g)]
    end
    return A
end

rand!(A::AbstractArray, r::AbstractArray) = rand!(GLOBAL_RNG, A, r)

rand(rng::AbstractRNG, r::AbstractArray{T}, dims::Dims) where {T} =
    rand!(rng, Array{T}(dims), r)
rand(                  r::AbstractArray,    dims::Dims) =
    rand(GLOBAL_RNG, r, dims)
rand(rng::AbstractRNG, r::AbstractArray,    dims::Integer...) =
    rand(rng, r, convert(Dims, dims))
rand(                  r::AbstractArray,    dims::Integer...) =
    rand(GLOBAL_RNG, r, convert(Dims, dims))

## random values from Dict, Set, IntSet

function rand(r::AbstractRNG, t::Dict)
    isempty(t) && throw(ArgumentError("collection must be non-empty"))
    rg = RangeGenerator(1:length(t.slots))
    while true
        i = rand(r, rg)
        Base.isslotfilled(t, i) && @inbounds return (t.keys[i] => t.vals[i])
    end
end

rand(r::AbstractRNG, s::Set) = rand(r, s.dict).first

function rand(r::AbstractRNG, s::IntSet)
    isempty(s) && throw(ArgumentError("collection must be non-empty"))
    # s can be empty while s.bits is not, so we cannot rely on the
    # length check in RangeGenerator below
    rg = RangeGenerator(1:length(s.bits))
    while true
        n = rand(r, rg)
        @inbounds b = s.bits[n]
        b && return n
    end
end

function nth(iter, n::Integer)::eltype(iter)
    for (i, x) in enumerate(iter)
        i == n && return x
    end
end
nth(iter::AbstractArray, n::Integer) = iter[n]

rand(r::AbstractRNG, s::Union{Associative,AbstractSet}) = nth(s, rand(r, 1:length(s)))

rand(s::Union{Associative,AbstractSet}) = rand(GLOBAL_RNG, s)

### arrays

function rand!(r::AbstractRNG, A::AbstractArray, s::Union{Dict,Set,IntSet})
    for i in eachindex(A)
        @inbounds A[i] = rand(r, s)
    end
    A
end

# avoid linear complexity for repeated calls with generic containers
rand!(r::AbstractRNG, A::AbstractArray, s::Union{Associative,AbstractSet}) =
    rand!(r, A, collect(s))

rand!(A::AbstractArray, s::Union{Associative,AbstractSet}) = rand!(GLOBAL_RNG, A, s)

rand(r::AbstractRNG, s::Associative{K,V}, dims::Dims) where {K,V} =
    rand!(r, Array{Pair{K,V}}(dims), s)
rand(r::AbstractRNG, s::AbstractSet{T}, dims::Dims) where {T} = rand!(r, Array{T}(dims), s)
rand(r::AbstractRNG, s::Union{Associative,AbstractSet}, dims::Integer...) =
    rand(r, s, convert(Dims, dims))
rand(s::Union{Associative,AbstractSet}, dims::Integer...) =
    rand(GLOBAL_RNG, s, convert(Dims, dims))
rand(s::Union{Associative,AbstractSet}, dims::Dims) = rand(GLOBAL_RNG, s, dims)

## random characters from a string

isvalid_unsafe(s::String, i) = !Base.is_valid_continuation(unsafe_load(pointer(s), i))
isvalid_unsafe(s::AbstractString, i) = isvalid(s, i)
_endof(s::String) = sizeof(s)
_endof(s::AbstractString) = endof(s)

function rand(rng::AbstractRNG, s::AbstractString)::Char
    g = RangeGenerator(1:_endof(s))
    while true
        pos = rand(rng, g)
        isvalid_unsafe(s, pos) && return s[pos]
    end
end

rand(s::AbstractString) = rand(GLOBAL_RNG, s)

### arrays

# we use collect(str), which is most of the time more efficient than specialized methods
# (except maybe for very small arrays)
rand!(rng::AbstractRNG, A::AbstractArray, str::AbstractString) = rand!(rng, A, collect(str))
rand!(A::AbstractArray, str::AbstractString) = rand!(GLOBAL_RNG, A, str)
rand(rng::AbstractRNG, str::AbstractString, dims::Dims) =
    rand!(rng, Array{eltype(str)}(dims), str)
rand(rng::AbstractRNG, str::AbstractString, d1::Integer, dims::Integer...) =
    rand(rng, str, convert(Dims, tuple(d1, dims...)))
rand(str::AbstractString, dims::Dims) = rand(GLOBAL_RNG, str, dims)
rand(str::AbstractString, d1::Integer, dims::Integer...) =
    rand(GLOBAL_RNG, str, d1, dims...)

## randstring (often useful for temporary filenames/dirnames)

"""
    randstring([rng=GLOBAL_RNG], [chars], [len=8])

Create a random string of length `len`, consisting of characters from
`chars`, which defaults to the set of upper- and lower-case letters
and the digits 0-9. The optional `rng` argument specifies a random
number generator, see [Random Numbers](@ref).

# Examples
```jldoctest
julia> srand(0); randstring()
"c03rgKi1"

julia> randstring(MersenneTwister(0), 'a':'z', 6)
"wijzek"

julia> randstring("ACGT")
"TATCGGTC"
```

!!! note
    `chars` can be any collection of characters, of type `Char` or
    `UInt8` (more efficient), provided [`rand`](@ref) can randomly
    pick characters from it.
"""
function randstring end

let b = UInt8['0':'9';'A':'Z';'a':'z']
    global randstring
    randstring(r::AbstractRNG, chars=b, n::Integer=8) = String(rand(r, chars, n))
    randstring(r::AbstractRNG, n::Integer) = randstring(r, b, n)
    randstring(chars=b, n::Integer=8) = randstring(GLOBAL_RNG, chars, n)
    randstring(n::Integer) = randstring(GLOBAL_RNG, b, n)
end
