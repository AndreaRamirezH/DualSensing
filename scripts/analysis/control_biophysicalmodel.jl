"""
STG136 - ramp input variance
Integral control with u = Δgsyn 
"""

function uncontrolled(neuronid, Bank, x0, input, tspan_long)
    conductances = Vector(Bank[neuronid, :])
    p = (conductances, input, dt)
    init_cond = copy(u0)
    init_cond[3] = x0
    prob_ = ODEProblem(neuron_ode!, init_cond, tspan_long, p)
    sol_ = OrdinaryDiffEq.solve(prob_, Tsit5(), dt=dt;
        save_idxs=[1, 2, 3], saveat=tspan_long[1]:dt:tspan_long[2], maxiters=1e12)
    v_traces = Float32.(sol_[1, :])
    #return v_traces, Float32(sol_[3, end])
    return sol_
end
function control_gsyn_ode!(du, u, p, t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = p[1]
    inp = p[2]
    dt = p[3]
    τ_g = p[4]
    σ2⁺ = p[5]
    ϵ2, β = p[6]
    γ = p[7]

    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v, ca, avg = u[1:3]
    x = u[4:14]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x
    Δgsyn = u[15]

    du[1] = sum([(gNa) * mNa^3 * hNa^1 * (E_Na - v),
        (gCaS) * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
        (gCaT) * mCaT^3 * hCaT^1 * (DynECa(ca) - v),
        (gKa) * mKa^3 * hKa^1 * (E_K - v),
        (gKCa) * mKCa^4 * (E_K - v),
        (gKd ) * mKd^4 * (E_K - v),
        (gH) * mH * (E_H - v),
        (gleak) * (E_L - v),
        (1+ Δgsyn)*inp[timeidx]
    ]) / Cm

    tmpICa = (gCaS) * mCaS^3 * hCaS * (DynECa(ca) - v) +
             (gCaT) * mCaT^3 * hCaT * (DynECa(ca) - v)

    du[2] = (1 / τCa) * (-ca + 0.05 + factorarea * tmpICa / Cm)
    du[3] = (1 / τav) * (-avg + ca)
    du[4:14] .= A_stg(v) * (B_stg(v, ca) - x)
    du[15] = (γ /τ_g) * ((σ2⁺-ϵ2) - β * avg) / σ2⁺
end


neuronid = 136
conductances = Vector(ActiveBank[neuronid, :])
model_XY1 = linearmodel_XY1(list_stgvariance_dataframes, neuronid)
model_XY2 = linearmodel_XY2(list_stgvariance_dataframes, neuronid)
ϵ1, α = coef(model_XY1)
ϵ2, β = coef(model_XY2)
ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α #function that goes from mean to variance

τ_g = 10 * τav # time constant for control variables
γ = 0.2 
μᵢ = 0.0f0
#need to create a pink noise input where input variance is ramped
number_changes = 12
n = length(tspan[1]:dt:number_changes * τ_g)
σ2ramp = range(noise_levels[2],noise_levels[1+number_changes],length=n)
input = ramp_pink_noise(n, μᵢ, σ2ramp.^2)

tspan_long = (tspan[1], number_changes * τ_g)
time_vec_long = tspan_long[1]:dt:tspan_long[2]
σᵢ = 1.0f0 # initial noise level
start_μ = mean(list_stgvariance_dataframes[findfirst(x -> x == σᵢ, noise_levels)][neuronid, :meanR])
start_σ2 = mean(list_stgvariance_dataframes[findfirst(x -> x == σᵢ, noise_levels)][neuronid, :σ2ᵣ])
start_x = mean(list_stgvariance_dataframes[findfirst(x -> x == σᵢ, noise_levels)][neuronid, :avgCa])

#sol_free = uncontrolled(neuronid, ActiveBank, start_x, input, tspan_long)

"""
Now add control of conductances 
"""
μ⁺ = 23.0f0
σ2⁺ = ĥ₋(μ⁺, α, β, ϵ1, ϵ2) #variance target based on mean target
target_acc = (μ⁺ - ϵ1) / α #target for average calcium

init_cond = [u0[1],u0[2], start_x, u0[4:end]..., 0.0]
p = (conductances, input, dt, τ_g, σ2⁺, (ϵ2, β), γ)
prob = ODEProblem(control_gsyn_ode!, init_cond, tspan_long, p)
sol_full = OrdinaryDiffEq.solve(prob, TRBDF2(), dt=dt; saveat=time_vec_long, maxiters=1e12,)


running_stats = compute_running_ratestats(sol_full, τ_g)
#replace!(running_stats[2], NaN => 0.0)
RMSEs = (sqrt(mean((running_stats[1] .- μ⁺).^2)),
        sqrt(mean((running_stats[2] .- σ2⁺).^2)))
goodness = RMSEs ./ [μ⁺, σ2⁺]


#save
jldsave("data/raw/STG/control/biophysical_gsyn_rampvariance.jld2"; 
sol_full=sol_full, p=p, running_stats=running_stats, RMSEs=RMSEs, sol_free=sol_free)
