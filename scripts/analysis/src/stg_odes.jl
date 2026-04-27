function sigmoid(V::R, vhalf::T, slope::T) where {R, T <: Real}
    one(R) / (one(R) + exp((V + vhalf) / slope))
end

function DynECa(Ca::T) where {T<:Real}
    k1 = T(500.0) * T(8.6174e-5) * T(283.15)
    cmax = T(3000.0) / Ca
    cmin = T(0.001)
    cval = ifelse(cmax > cmin, cmax, cmin)
    return k1 * log(cval)
end

m∞Na(V) = sigmoid(V, 25.5f0, -5.29f0)
h∞Na(V) = sigmoid(V, 48.9f0, 5.18f0)
τmNa(V) = 1.32f0 - 1.26f0 / (1.0f0 + exp((V + 120.0f0) / -25.0f0))
τhNa(V) = (0.67f0 / (1.0f0 + exp((V + 62.9f0) / -10.0f0))) *
                   (1.5f0 + 1.0f0 / (1.0f0 + exp((V + 34.9f0) / 3.6f0)))

m∞CaS(V) = sigmoid(V, 33.0f0, -8.1f0)
h∞CaS(V) = sigmoid(V, 60.0f0, 6.2f0)
τmCaS(V) = 1.4f0 + 7.0f0 / (exp((V + 27.0f0) / 10.0f0) + exp((V + 70.0f0) / -13.0f0))
τhCaS(V) = 60.0f0 + 150.0f0 / (exp((V + 55.0f0) / 9.0f0) + exp((V + 65.0f0) / -16.0f0))

m∞CaT(V) = sigmoid(V, 27.1f0, -7.2f0)
h∞CaT(V) = sigmoid(V, 32.1f0, 5.5f0)
τmCaT(V) = 21.7f0 - 21.3f0 / (1.0f0 + exp((V + 68.1f0) / -20.5f0))
τhCaT(V) = 105.0f0 - 89.8f0 / (1.0f0 + exp((V + 55.0f0) / -16.9f0))

m∞Kdr(V) = sigmoid(V, 12.3f0, -11.8f0)
τmKdr(V) = 7.2f0 - 6.4f0 / (1.0f0 + exp((V + 28.3f0) / -19.2f0))
h∞Kdr(V) = 1.0f0
τhKdr(V) = 1.0f0

m∞Ka(V) = sigmoid(V, 27.2f0, -8.7f0)
h∞Ka(V) = sigmoid(V, 56.9f0, 4.9f0)
τmKa(V) = 11.6f0 - 10.4f0 / (1.0f0 + exp((V + 32.9f0) / -15.2f0))
τhKa(V) = 38.6f0 - 29.2f0 / (1.0f0 + exp((V + 38.9f0) / -26.5f0))

m∞KCa(V::R, Ca::T) where {R,T<:Real} = (Ca / (Ca + 3.0f0)) * sigmoid(V, 28.3f0, -12.6f0)
τmKCa(V) = 90.3f0 - 75.1f0 / (1.0f0 + exp((V + 46.0f0) / -22.7f0))
h∞KCa(V) = 1.0f0
τhKCa(V) = 1.0f0

m∞H(V) = sigmoid(V, 70.0f0, 6.0f0)
τmH(V) = 272.0f0 + 1499.0f0 / (1.0f0 + exp((V + 42.2f0) / -8.73f0))
h∞H(V) = 1.0f0
τhH(V) = 1.0f0

const Cm = 10.0f0 # specific capacitance cₘ is a biological constant ~ 10 nF/mm^2
const τCa = 20.0f0 #20 ms
const factorarea = 14.96f0 * 0.0628f0 #from Liu et al '98
const τav = 6000.0f0

const E_Na = 50.0f0
const E_K = -80.0f0
const E_L = -50.0f0
const E_H = -20.0f0
const E_Ca = 85.0f0 #for reference purely

A_stg(v) = diagm(1 ./ [τmNa(v), τhNa(v), τmCaS(v), τhCaS(v), τmCaT(v), τhCaT(v), τmKa(v), τhKa(v), τmKCa(v), τmKdr(v), τmH(v)])
B_stg(v, ca) = [m∞Na(v), h∞Na(v), m∞CaS(v), h∞CaS(v), m∞CaT(v), h∞CaT(v), m∞Ka(v), h∞Ka(v), m∞KCa(v, ca), m∞Kdr(v), m∞H(v)]

function neuron_ode!(du,u,p,t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = p[1]
    inp = p[2]
    dt = p[3]
    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v,ca,avg = u[1:3]
    x = u[4:end]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x

    du[1] = sum([gNa * mNa^3 * hNa^1 * (E_Na - v),
            gCaS * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
            gCaT * mCaT^3 * hCaT^1 * (DynECa(ca) - v),
            gKa * mKa^3 * hKa^1 * (E_K - v),
            gKCa * mKCa^4 * (E_K - v),
            gKd * mKd^4 * (E_K - v),
            gH * mH * (E_H - v),
            gleak *(E_L - v),
            1 * inp[timeidx]
            ]) /Cm

    du[2] = (1 /τCa) * (-ca + 0.05 + (factorarea * sum([gCaS * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
                                                    gCaT * mCaT^3 * hCaT^1 * (DynECa(ca) - v)]) / Cm))
    du[3] = (1 / τav) * (-avg + ca)

    du[4:end] .= A_stg(v) * (B_stg(v,ca) - x)
end

function neurons_ode!(du, u, p, t)
    conductances = p[1]
    inputs = p[2]
    dt = p[3]
    num_trials = length(inputs)

    #call neuron_ode
    for i = 1:num_trials
        dz = @view du[1+(i-1)*14:14*i]
        z = @view u[1+(i-1)*14:14*i]
        neuron_ode!(dz, z, (conductances, inputs[i], dt), t)
    end

end

function neuron_syn_ode!(du, u, p, t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak, gsyn = p[1]
    inp = p[2]
    dt = p[3]
    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v, ca, avg = u[1:3]
    x = u[4:end]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x

    du[1] = sum([gNa * mNa^3 * hNa^1 * (E_Na - v),
        gCaS * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
        gCaT * mCaT^3 * hCaT^1 * (DynECa(ca) - v),
        gKa * mKa^3 * hKa^1 * (E_K - v),
        gKCa * mKCa^4 * (E_K - v),
        gKd * mKd^4 * (E_K - v),
        gH * mH * (E_H - v),
        gleak * (E_L - v),
        gsyn * inp[timeidx]
    ]) / Cm

    du[2] = (1 / τCa) * (-ca + 0.05 + (factorarea * sum([gCaS * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
                             gCaT * mCaT^3 * hCaT^1 * (DynECa(ca) - v)]) / Cm))
    du[3] = (1 / τav) * (-avg + ca)

    du[4:end] .= A_stg(v) * (B_stg(v, ca) - x)
end

function neurons_syn_ode!(du, u, p, t)
    conductances = p[1]
    inputs = p[2]
    dt = p[3]
    num_trials = length(inputs)

    #call neuron_ode
    for i = 1:num_trials
        dz = @view du[1+(i-1)*14:14*i]
        z = @view u[1+(i-1)*14:14*i]
        neuron_syn_ode!(dz, z, (conductances, inputs[i], dt), t)
    end

end

u0 = [-60.0f0, 0.05f0, 1.0f0, B_stg(-60.0f0, 0.05f0)...]; #initial conditions for a single neuron