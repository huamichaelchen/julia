# This file is a part of Julia. License is MIT: https://julialang.org/license

# Generate random integer within a range

abstract type RangeGenerator end

## RangeGenerator for BitInteger

# remainder function according to Knuth, where rem_knuth(a, 0) = a
rem_knuth(a::UInt, b::UInt) = a % (b + (b == 0)) + a * (b == 0)
rem_knuth(a::T, b::T) where {T<:Unsigned} = b != 0 ? a % b : a

# maximum multiple of k <= 2^bits(T) decremented by one,
# that is 0xFFFF...FFFF if k = typemax(T) - typemin(T) with intentional underflow
# see http://stackoverflow.com/questions/29182036/integer-arithmetic-add-1-to-uint-max-and-divide-by-n-without-overflow
maxmultiple(k::T) where {T<:Unsigned} = (div(typemax(T) - k + oneunit(k), k + (k == 0))*k + k - oneunit(k))::T

# maximum multiple of k within 1:2^32 or 1:2^64 decremented by one, depending on size
maxmultiplemix(k::UInt64) = if k >> 32 != 0; maxmultiple(k); else (div(0x0000000100000000, k + (k == 0))*k - oneunit(k))::UInt64; end

struct RangeGeneratorInt{T<:Integer,U<:Unsigned} <: RangeGenerator
    a::T   # first element of the range
    k::U   # range length or zero for full range
    u::U   # rejection threshold
end
# generators with 32, 128 bits entropy
RangeGeneratorInt(a::T, k::U) where {T,U<:Union{UInt32,UInt128}} = RangeGeneratorInt{T,U}(a, k, maxmultiple(k))
# mixed 32/64 bits entropy generator
RangeGeneratorInt(a::T, k::UInt64) where {T} = RangeGeneratorInt{T,UInt64}(a, k, maxmultiplemix(k))

function RangeGenerator(r::UnitRange{T}) where T<:Unsigned
    isempty(r) && throw(ArgumentError("range must be non-empty"))
    RangeGeneratorInt(first(r), last(r) - first(r) + oneunit(T))
end

for (T, U) in [(UInt8, UInt32), (UInt16, UInt32),
               (Int8, UInt32), (Int16, UInt32), (Int32, UInt32), (Int64, UInt64), (Int128, UInt128),
               (Bool, UInt32)]

    @eval RangeGenerator(r::UnitRange{$T}) = begin
        if isempty(r)
            throw(ArgumentError("range must be non-empty"))
        end
        RangeGeneratorInt(first(r), convert($U, unsigned(last(r) - first(r)) + one($U))) # overflow ok
    end
end

## RangeGenerator for BigInt

struct RangeGeneratorBigInt <: RangeGenerator
    a::BigInt         # first
    m::BigInt         # range length - 1
    nlimbs::Int       # number of limbs in generated BigInt's (z ∈ [0, m])
    nlimbsmax::Int    # max number of limbs for z+a
    mask::Limb        # applied to the highest limb
end


function RangeGenerator(r::UnitRange{BigInt})
    m = last(r) - first(r)
    m < 0 && throw(ArgumentError("range must be non-empty"))
    nd = ndigits(m, 2)
    nlimbs, highbits = divrem(nd, 8*sizeof(Limb))
    highbits > 0 && (nlimbs += 1)
    mask = highbits == 0 ? ~zero(Limb) : one(Limb)<<highbits - one(Limb)
    nlimbsmax = max(nlimbs, abs(last(r).size), abs(first(r).size))
    return RangeGeneratorBigInt(first(r), m, nlimbs, nlimbsmax, mask)
end

## rand(::RangeGenerator)

# this function uses 32 bit entropy for small ranges of length <= typemax(UInt32) + 1
# RangeGeneratorInt is responsible for providing the right value of k
function rand(rng::AbstractRNG, g::RangeGeneratorInt{T,UInt64}) where T<:Union{UInt64,Int64}
    local x::UInt64
    if (g.k - 1) >> 32 == 0
        x = rand(rng, UInt32)
        while x > g.u
            x = rand(rng, UInt32)
        end
    else
        x = rand(rng, UInt64)
        while x > g.u
            x = rand(rng, UInt64)
        end
    end
    return reinterpret(T, reinterpret(UInt64, g.a) + rem_knuth(x, g.k))
end

function rand(rng::AbstractRNG, g::RangeGeneratorInt{T,U}) where U<:Unsigned where T<:Integer
    x = rand(rng, U)
    while x > g.u
        x = rand(rng, U)
    end
    (unsigned(g.a) + rem_knuth(x, g.k)) % T
end

function rand(rng::AbstractRNG, g::RangeGeneratorBigInt)
    x = MPZ.realloc2(g.nlimbsmax*8*sizeof(Limb))
    limbs = unsafe_wrap(Array, x.d, g.nlimbs)
    while true
        rand!(rng, limbs)
        @inbounds limbs[end] &= g.mask
        MPZ.mpn_cmp(x, g.m, g.nlimbs) <= 0 && break
    end
    # adjust x.size (normally done by mpz_limbs_finish, in GMP version >= 6)
    x.size = g.nlimbs
    while x.size > 0
        @inbounds limbs[x.size] != 0 && break
        x.size -= 1
    end
    MPZ.add!(x, g.a)
end

### arrays

function rand!(rng::AbstractRNG, A::AbstractArray, g::RangeGenerator)
    for i in eachindex(A)
        @inbounds A[i] = rand(rng, g)
    end
    return A
end

## rand(::UnitRange)

rand(rng::AbstractRNG, r::UnitRange{<:Union{Signed,Unsigned,BigInt,Bool}}) = rand(rng, RangeGenerator(r))

rand!(rng::AbstractRNG, A::AbstractArray, r::UnitRange{<:Union{Signed,Unsigned,BigInt,Bool,Char}}) = rand!(rng, A, RangeGenerator(r))
