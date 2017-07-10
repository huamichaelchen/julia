# This file is a part of Julia. License is MIT: https://julialang.org/license

## RandomDevice

const BitIntegerType = Union{map(T->Type{T}, Base.BitInteger_types)...}
const BoolBitIntegerType = Union{Type{Bool},BitIntegerType}
const BoolBitIntegerArray = Union{Array{Bool},Base.BitIntegerArray}

if is_windows()
    struct RandomDevice <: AbstractRNG
        buffer::Vector{UInt128}

        RandomDevice() = new(Vector{UInt128}(1))
    end

    function rand(rd::RandomDevice, T::BoolBitIntegerType)
        win32_SystemFunction036!(rd.buffer)
        @inbounds return rd.buffer[1] % T
    end

    rand!(rd::RandomDevice, A::BoolBitIntegerArray) = (win32_SystemFunction036!(A); A)
else # !windows
    struct RandomDevice <: AbstractRNG
        file::IOStream
        unlimited::Bool

        RandomDevice(unlimited::Bool=true) = new(open(unlimited ? "/dev/urandom" : "/dev/random"), unlimited)
    end

    rand(rd::RandomDevice, T::BoolBitIntegerType)   = read( rd.file, T)
    rand!(rd::RandomDevice, A::BoolBitIntegerArray) = read!(rd.file, A)
end # os-test

"""
    RandomDevice()

Create a `RandomDevice` RNG object. Two such objects will always generate different streams of random numbers.
"""
RandomDevice


rand(rng::RandomDevice, ::Type{Close1Open2}) =
    reinterpret(Float64, 0x3ff0000000000000 | rand(rng, UInt64) & 0x000fffffffffffff)

rand(rng::RandomDevice, ::Type{CloseOpen}) = rand(rng, Close1Open2) - 1.0
