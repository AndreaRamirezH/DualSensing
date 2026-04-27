abstract type IonSpecies end
struct Sodium <: IonSpecies end
struct Potassium <: IonSpecies end
struct Calcium <: IonSpecies end
struct Proton <: IonSpecies end
struct Leak <: IonSpecies end
struct External <: IonSpecies end

reversals = Dict{DataType,Float32}(
    Sodium => 50.0,
    Potassium => -80.0,
    Leak => -50.0,
    Proton => -20.0,
    Calcium => 85.0)

#################### NaV ###############################
struct Na{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
Na(x) = Na(x, 3, 1, Union{}, Sodium)
m∞(::Na, V::T) where {T<:Real} = m∞Na(V)
h∞(::Na, V::T) where {T<:Real} = h∞Na(V)
τm(::Na, V::T) where {T<:Real} = τmNa(V)
τh(::Na, V::T) where {T<:Real} = τhNa(V)


#################### Slow calcium current #############################
struct CaS{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
CaS(x) = CaS(x, 3, 1, Union{}, Calcium)
m∞(::CaS, V::T) where {T<:Real} = m∞CaS(V)
h∞(::CaS, V::T) where {T<:Real} = h∞CaS(V)
τm(::CaS, V::T) where {T<:Real} = τmCaS(V)
τh(::CaS, V::T) where {T<:Real} = τhCaS(V)

#################### Transient calcium current ######################
struct CaT{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
CaT(x) = CaT(x, 3, 1, Union{}, Calcium)
m∞(::CaT, V::T) where {T<:Real} = m∞CaT(V)
h∞(::CaT, V::T) where {T<:Real} = h∞CaT(V)
τm(::CaT, V::T) where {T<:Real} = τmCaT(V)
τh(::CaT, V::T) where {T<:Real} = τhCaT(V)

#################### A-type potassium current #########################
struct Ka{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
Ka(x) = Ka(x, 3, 1, Union{}, Potassium)
m∞(::Ka, V::T) where {T<:Real} = m∞Ka(V)
h∞(::Ka, V::T) where {T<:Real} = h∞Ka(V)
τm(::Ka, V::T) where {T<:Real} = τmKa(V)
τh(::Ka, V::T) where {T<:Real} = τhKa(V)


################### Calcium-activated potassium current ########
struct KCa{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
KCa(x) = KCa(x, 4, 0, Calcium, Potassium)
m∞(::KCa, V::T, Ca::S) where {T,S<:Real} = m∞KCa(V, Ca)
τm(::KCa, V::T) where {T<:Real} = τmKCa(V)
h∞(::KCa, V::T) where {T<:Real} = 1.0
τh(::KCa, V::T) where {T<:Real} = 1.0


#################### Delayed rectifier potassium current ######################
struct Kdr{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
Kdr(x) = Kdr(x, 4, 0, Union{}, Potassium)
m∞(::Kdr, V::T) where {T<:Real} = m∞Kdr(V)
τm(::Kdr, V::T) where {T<:Real} = τmKdr(V)
h∞(::Kdr, V::T) where {T<:Real} = 1.0
τh(::Kdr, V::T) where {T<:Real} = 1.0

#################### H current ####################
struct H{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    p::Int64
    q::Int64
    sensed::Type{S}
    actuated::Type{A}
end
H(x) = H(x, 1, 0, Union{}, Proton)
m∞(::H, V::T) where {T<:Real} = m∞H(V)
τm(::H, V::T) where {T<:Real} = τmH(V)
h∞(::H, V::T) where {T<:Real} = 1.0
τh(::H, V::T) where {T<:Real} = 1.0

##################### leak ######################
struct leak{C<:AbstractVector,S<:IonSpecies,A<:IonSpecies}
    g::C
    sensed::Type{S}
    actuated::Type{A}
end

leak(x) = leak(x, Union{}, Leak)

function I∞(ch::Union{Na, CaS, CaT, Ka, KCa, Kdr}, V::T) where {T<:Real}
    first(ch.g) * m∞(ch, V)^ch.p * h∞(ch, V)^ch.q * (reversals[ch.actuated] - V)
end

function I∞(ch::KCa, V::T, channels::Tuple) where {T<:Real}
    first(ch.g) * m∞(ch, V, Ca_∞(V, channels))^ch.p * h∞(ch, V)^ch.q * (reversals[ch.actuated] - V)
end

function I∞(ch::leak, V::Float32)
    first(ch.g) * (reversals[ch.actuated] - V)
end

check_biophysics(params::Vector{T}) where {T<:Real} = prod(params .> 0)
get_gs(channels::Tuple) = Vector{Float32}([first(ch.g) for ch in channels])
ionic(channels::Tuple) = collect(ch for ch in channels if ch.actuated != External && ch.actuated != Leak) |> Tuple

function make_channels(conductances::NTuple{8,T}) where {T<:Real}
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = conductances
    return (Na([gNa]), CaS([gCaS]), CaT([gCaT]), Ka([gKa]),
        KCa([gKCa]), Kdr([gKd]), H([gH]), leak([gleak]))
end

"""
    function DICweights(τX::Function, taus::Tuple, V::Float32)

Computes ratios of log distances from a timeconstant to characteristic timescales at a given voltage
W_fs = log(dist(τX, τs)) / log(dist(τX, τf))
W_su = log(dist(τX, τu)) / log(dist(τX, τs))
"""
function DICweights(τX::Function, V::Float32) 
    τf, τs, τu = τmNa, τmKdr, τmH #τf, τs, τu are the fast, slow and ultraslow timescale

    if τX(V) <= τf(V)
        W_fs = 1.0
        W_su = 1.0
    elseif τf(V) <= τX(V) <= τs(V)
        W_fs = (log(τs(V)) - log(τX(V))) / (log(τs(V)) - log(τf(V)))
        W_su = 1.0
    elseif τs(V) <= τX(V) <= τu(V)
        W_fs = 0.0
        W_su = (log(τu(V)) - log(τX(V))) / (log(τu(V)) - log(τs(V)))
    elseif τu(V) < τX(V)
        W_fs = 0.0
        W_su = 0.0
    end
    return Float32(W_fs), Float32(W_su)
end

function which_dic(timescale::Integer, weights::Tuple{Float32,Float32})
    W_fs, W_su = weights
    cases = Dict(
        1 => W_fs,
        2 => W_su - W_fs,
        3 => 1 - W_su
    )
    return cases[timescale]
end

function Ca_∞(V::T, channels::Tuple) where {T<:Real}
    #[Ca](V) = Ca∞ + (calc_multiplier * ca_currents(V)) / Cₘ) when D(Ca)=0, so at steady state
    ca_channels = filter(ch -> Calcium == ch.actuated, channels)
    return 0.05 + (factorarea * sum(I∞(ch, V) for ch in ca_channels)) / Cm
end

"""
    function Gate_properties(channel::IonChannel)

Returns a 2-element vector of the channel's gating characteristics which determine its dynamics,
i.e. the functions τ_gate_ and _gate_∞ for activating and inactivating gates.
"""
function Gate_properties(ch::Union{Na, CaS, CaT, Ka, Kdr, H}, channels::Tuple)
    [(V -> τm(ch, V), V -> m∞(ch, V)),
     (V -> τh(ch, V), V -> h∞(ch, V))]
end

function Gate_properties(ch::KCa, channels::Tuple)
    [(V -> τm(ch, V), V -> m∞(ch, V, Ca_∞(-50.0, channels))),
     (V -> τh(ch, V), V -> h∞(ch, V))]
end

"""
    function ∇I∞_dXinf(ch::IonChannel, b::Neuron, Veval::Float32, normalize::Bool=false)

Returns the symbolic differentiation of steady state current I∞ for a given channel
    with respect to both gating variables Xinf = m∞ or h∞, evaluated at Veval.
"""

function ∇I∞_dXinf(ch::Union{Na, CaS, CaT, Ka, KCa, Kdr, H}, channels::Tuple, Veval::Float32, normalize::Bool=false)
    g = normalize ? 1.0 : first(ch.g)
    m = Calcium == ch.sensed ? m∞(ch, Veval, Ca_∞(Veval, channels)) : m∞(ch, Veval)
    ∇I∞_dXinfs = [g * ch.p * m^(ch.p - 1) * h∞(ch, Veval)^ch.q * (reversals[ch.actuated] - Veval),
        g * m^ch.p * ch.q * h∞(ch, Veval)^(ch.q - 1) * (reversals[ch.actuated] - Veval)]
end

dXinf_dV(Xinf::Function, V::T) where {T<:Real} = ForwardDiff.derivative(V -> Xinf(V), V) |> Symbolics.value

function Aux(tup::Tuple{Function,Function}, timescale::Int64, V::Float32)
    τX, X∞ = tup
    return which_dic(timescale, DICweights(τX, V)) * dXinf_dV(X∞, V)
end

"""
    function addend(channel::IonChannel, b::BasicNeuron, timescale::Int64, V::Float, normalize::Bool)

Returns the contribution of one channel to a DIC. Timescales: 1=>fast, 2=>slow, 3=>ultraslow.
Normalize argument signals if the conductance should be set to 1.0, used for the sensitivity matrix
"""

function Addend(channel::Union{Na, CaS, CaT, Ka, Kdr, H}, channels::Tuple, timescale::Int64, V::Float32, normalize::Bool)::Float32

    aux = map(tup -> Aux(tup, timescale, V), Gate_properties(channel, channels))
    dot(aux, ∇I∞_dXinf(channel, channels, V, normalize))
end

function Addend(channel::KCa, channels::Tuple, timescale::Int64, V::Float32, normalize::Bool)::Float32

    aux = map(tup -> Aux(tup, timescale, V), Gate_properties(channel, channels))
    res = dot(aux, ∇I∞_dXinf(channel, channels, V, normalize))
    if timescale == 3
        return res + addend_KCa_Ca_dependent(channel, channels, timescale, V, Ca_∞(V, channels))
        #to calculate the total derivative of m_kca∞(v,c) = ∂m_kca∞/∂v + ∂m_kca∞/∂c * dc/dv
    else
        return res
    end
end

function addend_KCa_Ca_dependent(kca_channel::KCa, channels::Tuple, timescale::Int64, V::Float32, Ca_eval)
    u = first(kca_channel.g) * 4 * m∞KCa(V, Ca_eval)^3 * (reversals[kca_channel.actuated] - V)

    #take partial of mkca(v,c) wrt c 
    v = ForwardDiff.gradient(x -> m∞KCa(x...), [V, Ca_eval])[2] |> Symbolics.value
    # finally, chain rule on Ca_∞(V) 
    y = ForwardDiff.derivative(x -> Ca_∞(x, channels), V) |> Symbolics.value

    return which_dic(timescale, DICweights(τmKCa, V)) * u * v * y
end

"""
    function _timescale_DIC(b::BasicNeuron, V::Float32)

returns DIC curves (i.e. functions of V) for a neuron with a given set of channels.
g_leak is excluded because it's a fixed parameter in all neurons
"""
function fastDIC(channels::Tuple, V::Float32)
    ionchannels = filter(x -> !(Leak == x.actuated || External == x.actuated), channels)
    sum(Addend(channel, channels, 1, V, false) for channel in ionchannels)
end

function slowDIC(channels::Tuple, V::Float32)
    ionchannels = filter(x -> !(Leak == x.actuated || External == x.actuated), channels)
    sum(Addend(channel, channels, 2, V, false) for channel in ionchannels)
end

function ultraslowDIC(channels::Tuple, V::Float32)
    ionchannels = filter(x -> !(Leak == x.actuated || External == x.actuated), channels)
    sum(Addend(channel, channels, 3, V, false) for channel in ionchannels)
end

function inputconductance(channels, V)
    gin = []
    for ch in ionic(channels)
        aux = [dXinf_dV(X∞, V) for (tau, X∞) in Gate_properties(ch, channels)]
        m = (Calcium == ch.sensed) ? m∞(ch, V, Ca_∞(V, channels)) : m∞(ch, V)
        push!(gin, dot(aux, ∇I∞_dXinf(ch, channels, V, false)) + first(ch.g) * m^ch.p * h∞(ch, V)^ch.q)
    end
    gleak = get_gs(filter(x -> Leak == x.actuated, channels))[1]
    return sum(gin) + gleak
end
