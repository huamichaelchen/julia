# This file is a part of Julia. License is MIT: https://julialang.org/license

## make_seed()

# make_seed produces values of type Array{UInt32}, suitable for MersenneTwister seeding
function make_seed()
    try
        return rand(RandomDevice(), UInt32, 4)
    catch
        println(STDERR,
                "Entropy pool not available to seed RNG; using ad-hoc entropy sources.")
        seed = reinterpret(UInt64, time())
        seed = hash(seed, UInt64(getpid()))
        try
            seed = hash(seed, parse(UInt64,
                                    readstring(pipeline(`ifconfig`, `sha1sum`))[1:40], 16))
        end
        return make_seed(seed)
    end
end

function make_seed(n::Integer)
    n < 0 && throw(DomainError())
    seed = UInt32[]
    while true
        push!(seed, n & 0xffffffff)
        n >>= 32
        if n == 0
            return seed
        end
    end
end

## srand()

"""
    srand([rng=GLOBAL_RNG], seed) -> rng
    srand([rng=GLOBAL_RNG]) -> rng

Reseed the random number generator. If a `seed` is provided, the RNG will give a
reproducible sequence of numbers, otherwise Julia will get entropy from the system. For
`MersenneTwister`, the `seed` may be a non-negative integer or a vector of [`UInt32`](@ref)
integers. `RandomDevice` does not support seeding.

# Examples
```jldoctest
julia> srand(1234);

julia> x1 = rand(2)
2-element Array{Float64,1}:
 0.590845
 0.766797

julia> srand(1234);

julia> x2 = rand(2)
2-element Array{Float64,1}:
 0.590845
 0.766797

julia> x1 == x2
true
```
"""
srand(r::MersenneTwister) = srand(r, make_seed())
srand(r::MersenneTwister, n::Integer) = srand(r, make_seed(n))

function srand(r::MersenneTwister, seed::Vector{UInt32})
    copy!(resize!(r.seed, length(seed)), seed)
    dsfmt_init_by_array(r.state, r.seed)
    mt_setempty!(r)
    return r
end

### GLOBAL_RNG

function dsfmt_gv_srand()
    # Temporary fix for #8874 and #9124: update global RNG for Rmath
    dsfmt_gv_init_by_array(GLOBAL_RNG.seed+UInt32(1))
    return GLOBAL_RNG
end

function srand()
    srand(GLOBAL_RNG)
    dsfmt_gv_srand()
end

function srand(seed::Union{Integer,Vector{UInt32}})
    srand(GLOBAL_RNG, seed)
    dsfmt_gv_srand()
end
