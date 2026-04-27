"""
Compute cumulant and correlation for a given x=ACC.
"""

using Statistics, StatsBase

"""
    autocorr(x, τ)

Estimate raw autocorrelation R_xx(τ) = E[x(t) x(t+τ)] for a discrete signal x.
Does not normalize by variance and does not subtract mean.
"""
function autocorr(x::Vector, τ::Int)
    N = length(x)
    if τ >= 0
        return mean(x[1:(N-τ)] .* x[(1+τ):N])
    else
        return autocorr(x, -τ)
    end
end

"""
    moment4(x, τ1, τ2, θ1, θ2)

Estimate fourth-order moment:
M4(τ1, τ2, θ1, θ2) = E[x(t-τ1) x(t-τ2) x(t-θ1) x(t-θ2)]
"""
function moment4(x::Vector, τ1::Int, τ2::Int, θ1::Int, θ2::Int)
    N = length(x)
    maxlag = maximum([τ1, τ2, θ1, θ2])
    idx_range = (1+maxlag):N   # ensure valid indices
    vals = [x[t-τ1] * x[t-τ2] * x[t-θ1] * x[t-θ2] for t in idx_range]
    return mean(vals)
end

"""
    cumulant4(x, τ1, τ2, θ1, θ2)

Compute fourth-order cumulant C4 using the definition of M4
"""
function cumulant4(x::Vector, τ1::Int, τ2::Int, θ1::Int, θ2::Int)
    M4 = moment4(x, τ1, τ2, θ1, θ2)

    R1 = autocorr(x, τ1 - τ2) * autocorr(x, θ1 - θ2)
    R2 = autocorr(x, τ1 - θ1) * autocorr(x, τ2 - θ2)
    R3 = autocorr(x, τ1 - θ2) * autocorr(x, τ2 - θ1)

    return M4 - (R1 + R2 + R3)
end

"""
    cumulant4_normalized(x, τ1, τ2, θ1, θ2)

Compute normalized fourth-order cumulant:
C4_norm = C4 / σ^4 where σ² = var(x)
The cumulant depends on signal scale, so normalization helps comparative analysis.
"""
function cumulant4_normalized(x::Vector, τ1::Int, τ2::Int, θ1::Int, θ2::Int)
    C4 = cumulant4(x, τ1, τ2, θ1, θ2)

    σ² = mean(x .^ 2)   # variance estimate (unbiased denominator not needed here)
    return C4 / (σ²^2)
end

# ------------------------------------
"""
show that in stg model there is a lot of structure in acc, but less so in CS model
"""
function CSneuron_tau!(du,u,p,t)
    gNa, gCa, gKa, gKd, gleak = p[1]
    inp = p[2]
    dt = p[3]
    τav = p[4]
    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v,ca,avg = u[1:3]
    x = u[4:end]
    mNa, hNa, mCa, mKa, hKa, mKd = x

    du[1] = sum([gNa * mNa^3 * hNa^1 * (Eq_Na - v),
            gCa * mCa^2 * (Eq_Ca - v),
            gKa * mKa^3 * hKa^1 * (Eq_K - v),
            gKd * mKd^4 * (Eq_K - v),
            gleak *(Eq_L - v),
            inp[timeidx]
            ]) /Cm

    du[2] = (1 /τCa) * (-ca + 0.05 + (factorarea * (gCa * mCa^2 * (Eq_Ca - v)) / Cm))
    du[3] = (1 / τav) * (-avg + ca)

    du[4:end] .= A_cs(v) * (B_cs(v) - x)
end

function analyze_cs_ca(NeuronBank, neuronid, sigma_in)
    conductances = Tuple(NeuronBank[neuronid, :])
    input = pinknoise_input[(0.0f0, sigma_in)][3]
    p = (conductances, input, dt, τav)
    csprob = ODEProblem(CSneuron_tau!, u0_CS, (0.0f0, 15000.0f0), p)
    sol = OrdinaryDiffEq.solve(csprob, Tsit5(), dt=dt; save_idxs=[1, 2, 3], saveat=0:15000)

    x = sol[2, :]

    xplt = plot(x, ylabel="[Ca]", xlabel="", label="", color=:black, xticks = 0:5000:10000,
        xlims=(0, 11000), grid=:x, lw=1.3, yticks=0:0.2:0.4)
    Mx = maximum(x)
    t0 = findfirst(x -> x > 0.85*Mx, x)
    t0 = 4800
    xzoomed = plot(x[t0:t0+1000], ylabel="[Ca]", xlabel="t (ms)", label="", color=:black,
        lw=2, grid=:x, yticks=0:0.2:0.4)

    maxlag = 1000 #this is in miliseconds
    R = autocor(x, [τ for τ in 0:10:maxlag]) #this uses StatsBase.autocor, the normalized version for clarity
    Rplt = plot(0:10:maxlag, R, seriestype=:stem, xlabel="Lag τ (ms)", ylabel="Rₓₓ(τ)", 
            label="", color=:black, ylims=(-1.0, 1.0), yticks=-1:1:1,)

    C4s_even = Float64[]
    delays = 0:5:250
    for d in delays  # step size for lags
        τ1, τ2, θ1, θ2 = cumsum([0, d, d, d]) # example lags in ms
        push!(C4s_even, cumulant4_normalized(x, τ1, τ2, θ1, θ2))
    end
    C4_pair = Float64[]
    for d in delays  # step size for lags
        τ1, τ2, θ1, θ2 = 0, d, 0, d
        push!(C4_pair, cumulant4_normalized(x, τ1, τ2, θ1, θ2))
    end

    C4plt = plot(delays, C4s_even, xlabel="Lag τ (ms)", ylabel="C₄", label="(0,τ,2τ,3τ)", color=:black, lw=2,
        foreground_color_legend=nothing, background_color_legend=nothing, legend_column=2, legend=:top, grid=:x, yticks=0:2:4)
    C4plt = plot!(delays, C4_pair, xlabel="Lag τ (ms)", ylabel="C₄", label="(0,τ,0,τ)", color=:darkblue, lw=2,)
    C4plt = hline!(C4plt, [0], lw=0.5, ls=:dash, color=:gray, label="")

    plot(xplt, xzoomed, Rplt, C4plt, layout=(4, 1), left_margin=10mm, size=(800, 800), 
        legendfontsize=14, labelfontsize=18,
        xtickfontsize=14, ytickfontsize=14, plot_title="Neuron $(neuronid) with σ_in=$(sigma_in)")
end
savefig(analyze_cs_ca(SilentBank, 48, 8.0f0), "../figures/InputDriven/cumulants_neur48.pdf")


function neuron_tau!(du, u, p, t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = p[1]
    inp = p[2]
    dt = p[3]
    τav = p[4]
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
        inp[timeidx]
    ]) / Cm

    du[2] = (1 / τCa) * (-ca + 0.05 + (factorarea * sum([gCaS * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
                             gCaT * mCaT^3 * hCaT^1 * (DynECa(ca) - v)]) / Cm))
    du[3] = (1 / τav) * (-avg + ca)

    du[4:end] .= A_stg(v) * (B_stg(v, ca) - x)
end

function analyze_stg_ca(NeuronBank, neuronid, sigma_in)
    conductances = Tuple(NeuronBank[neuronid, :])
    input = pinknoise_input[(0.0f0, sigma_in)][1]

    p = (conductances, input, dt, τav)
    stgprob = ODEProblem(neuron_tau!, u0, (0.0f0, 15000.0f0), p)
    sol = OrdinaryDiffEq.solve(stgprob, Tsit5(), dt=dt; save_idxs=[1, 2, 3], saveat=0:15000)

    #take a chunck of the solution but skip the initial transient
    x = sol[2,Int(transient):end]

    xplt = plot(x, ylabel="[Ca]", xlabel="", label="", color=:black, xticks=0:5000:10000,
        xlims=(0, 11000), grid=:x, yticks=0:20:40, lw=1.3)

    t0 = findfirst(x -> x > 1, diff(x))
    xzoomed = plot(x[t0:t0+1000], ylabel="[Ca]", xlabel="t (ms)", label="", 
            yticks=[0,20], lw = 2, color=:black, grid=:x)

    maxlag = 1000 #this is in miliseconds
    R = autocor(x, [τ for τ in 0:10:maxlag]) #this uses StatsBase.autocor
    Rplt = plot(0:10:maxlag, R, seriestype=:stem, xlabel="Lag τ (ms)", ylabel="Rₓₓ(τ)",
            yticks=-1:1:1, label="", color=:black, ylims=(-1.0,1.0))
    

    C4s_even = Float64[]
    delays = 0:5:250
    for d in delays  # step size for lags
        τ1, τ2, θ1, θ2 = cumsum([0, d, d, d]) # example lags in ms
        push!(C4s_even, cumulant4_normalized(x, τ1, τ2, θ1, θ2))
    end
    C4_pair = Float64[]
    for d in delays  # step size for lags
        τ1, τ2, θ1, θ2 = 0, d, 0, d
        push!(C4_pair, cumulant4_normalized(x, τ1, τ2, θ1, θ2))
    end

    C4plt = plot(delays, C4s_even, xlabel="Lag τ (ms)", ylabel="C₄", label="(0,τ,2τ,3τ)", color=:black, lw=2,
        foreground_color_legend=nothing, background_color_legend=nothing, legend_column=2, legend=:top, grid=:x,)
    C4plt = plot!(delays, C4_pair, xlabel="Lag τ (ms)", ylabel="C₄", label="(0,τ,0,τ)", color=:darkblue, lw=2,)
    C4plt = hline!(C4plt, [0], lw=0.5, ls=:dash, color=:gray, label="", )

    plot(xplt, xzoomed, Rplt, C4plt, layout=(4, 1), left_margin=10mm, size=(800,800), 
        legendfontsize=14, labelfontsize=18,
        xtickfontsize=14, ytickfontsize=14,
        plot_title="Neuron $(neuronid) with σ_in=$(sigma_in)")
end
savefig(analyze_stg_ca(BursterBank, 3, 1.0f0), "../figures/STG/cumulants_neur3.pdf")

