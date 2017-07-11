# This file is a part of Julia. License is MIT: https://julialang.org/license

## GLOBAL_RNG fallback for all numbers

@inline rand(T::Type) = rand(GLOBAL_RNG, T)

## random floating point values

@inline rand(r::AbstractRNG=GLOBAL_RNG) = rand(r, CloseOpen)

### MersenneTwister & RandomDevice

#### Float64

@inline rand(r::Union{RandomDevice,MersenneTwister}, ::Type{Float64}) = rand(r, CloseOpen)

@inline rand(r::MersenneTwister, ::Type{I}) where {I<:FloatInterval} = (reserve_1(r); rand_inbounds(r, I))

rand(rng::RandomDevice, ::Type{Close1Open2}) =
    reinterpret(Float64, 0x3ff0000000000000 | rand(rng, UInt64) & 0x000fffffffffffff)

rand(rng::RandomDevice, ::Type{CloseOpen}) = rand(rng, Close1Open2) - 1.0

#### Float16 & Float32

rand(r::Union{RandomDevice,MersenneTwister}, ::Type{Float16}) =
    Float16(reinterpret(Float32, (rand_ui10_raw(r) % UInt32 << 13) & 0x007fe000 | 0x3f800000) - 1)

rand(r::Union{RandomDevice,MersenneTwister}, ::Type{Float32}) =
    reinterpret(Float32, rand_ui23_raw(r) % UInt32 & 0x007fffff | 0x3f800000) - 1

## random integers

### helper functions

rand_ui10_raw(r::AbstractRNG) = rand(r, UInt16)
rand_ui23_raw(r::AbstractRNG) = rand(r, UInt32)
@inline rand_ui52_raw(r::AbstractRNG) = reinterpret(UInt64, rand(r, Close1Open2))
@inline rand_ui52(r::AbstractRNG) = rand_ui52_raw(r) & 0x000fffffffffffff

### MersenneTwister

@inline rand(r::MersenneTwister, ::Type{T}) where {T<:Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32}} =
    rand_ui52_raw(r) % T

function rand(r::MersenneTwister, ::Type{UInt64})
    reserve(r, 2)
    rand_ui52_raw_inbounds(r) << 32 ⊻ rand_ui52_raw_inbounds(r)
end

function rand(r::MersenneTwister, ::Type{UInt128})
    reserve(r, 3)
    xor(rand_ui52_raw_inbounds(r) % UInt128 << 96,
        rand_ui52_raw_inbounds(r) % UInt128 << 48,
        rand_ui52_raw_inbounds(r))
end

rand(r::MersenneTwister, ::Type{Int64})   = reinterpret(Int64,  rand(r, UInt64))
rand(r::MersenneTwister, ::Type{Int128})  = reinterpret(Int128, rand(r, UInt128))

## random complex values

rand(r::AbstractRNG, ::Type{Complex{T}}) where {T<:Real} = complex(rand(r, T), rand(r, T))

## random Char values

# returns a random valid Unicode scalar value (i.e. 0 - 0xd7ff, 0xe000 - # 0x10ffff)
function rand(r::AbstractRNG, ::Type{Char})
    c = rand(r, 0x00000000:0x0010f7ff)
    (c < 0xd800) ? Char(c) : Char(c+0x800)
end
