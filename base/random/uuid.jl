# This file is a part of Julia. License is MIT: https://julialang.org/license

# random UUID generation

struct UUID
    value::UInt128

    UUID(u::UInt128) = new(u)
end

"""
    uuid1([rng::AbstractRNG=GLOBAL_RNG]) -> UUID

Generates a version 1 (time-based) universally unique identifier (UUID), as specified
by RFC 4122. Note that the Node ID is randomly generated (does not identify the host)
according to section 4.5 of the RFC.

# Examples

```jldoctest
julia> rng = MersenneTwister(1234);

julia> Base.Random.uuid1(rng)
2cc938da-5937-11e7-196e-0f4ef71aa64b
```
"""
function uuid1(rng::AbstractRNG=GLOBAL_RNG)
    u = rand(rng, UInt128)

    # mask off clock sequence and node
    u &= 0x00000000000000003fffffffffffffff

    # set the unicast/multicast bit and version
    u |= 0x00000000000010000000010000000000

    # 0x01b21dd213814000 is the number of 100 nanosecond intervals
    # between the UUID epoch and Unix epoch
    timestamp = round(UInt64, time() * 1e7) + 0x01b21dd213814000
    ts_low = timestamp & typemax(UInt32)
    ts_mid = (timestamp >> 32) & typemax(UInt16)
    ts_hi = (timestamp >> 48) & 0x0fff

    u |= UInt128(ts_low) << 96
    u |= UInt128(ts_mid) << 80
    u |= UInt128(ts_hi) << 64

    UUID(u)
end

"""
    uuid4([rng::AbstractRNG=GLOBAL_RNG]) -> UUID

Generates a version 4 (random or pseudo-random) universally unique identifier (UUID),
as specified by RFC 4122.

# Example
```jldoctest
julia> rng = MersenneTwister(1234);

julia> Base.Random.uuid4(rng)
82015f10-44cc-4827-996e-0f4ef71aa64b
```
"""
function uuid4(rng::AbstractRNG=GLOBAL_RNG)
    u = rand(rng, UInt128)
    u &= 0xffffffffffff0fff3fffffffffffffff
    u |= 0x00000000000040008000000000000000
    UUID(u)
end

"""
    uuid_version(u::UUID) -> Integer

Inspects the given UUID and returns its version (see RFC 4122).

# Example

```jldoctest
julia> rng = MersenneTwister(1234);

julia> Base.Random.uuid_version(Base.Random.uuid4(rng))
4
```
"""
uuid_version(u::UUID) = Int((u.value >> 76) & 0xf)

Base.convert(::Type{UInt128}, u::UUID) = u.value

function Base.convert(::Type{UUID}, s::AbstractString)
    s = lowercase(s)

    if !ismatch(r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", s)
        throw(ArgumentError("Malformed UUID string"))
    end

    u = UInt128(0)
    for i in [1:8; 10:13; 15:18; 20:23; 25:36]
        u <<= 4
        d = s[i]-'0'
        u |= 0xf & (d-39*(d>9))
    end
    return UUID(u)
end

function Base.repr(u::UUID)
    u = u.value
    a = Vector{UInt8}(36)
    for i = [36:-1:25; 23:-1:20; 18:-1:15; 13:-1:10; 8:-1:1]
        d = u & 0xf
        a[i] = '0'+d+39*(d>9)
        u >>= 4
    end
    a[[24,19,14,9]] = '-'

    return String(a)
end

Base.show(io::IO, u::UUID) = write(io, Base.repr(u))
