"""
Control with intrinsic conductances when disturbance is step in input variance 
Neuron is initialized with a warm-up period
"""

using JLD2, RecursiveArrayTools, DataFrames, GLM

function controller_ode!(du, u, p, t)
    g = p[1]                   
    ctrl_idx1, ctrl_idx2 = p[9]

    inp = p[2]
    dt = p[3]
    τ_g = p[4]
    μ⁺, σ2⁺ = p[5]
    ϵ1, α = p[6]
    ϵ2, β = p[7]
    γ = p[8]

    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v, ca, avg = u[1:3]
    x = u[4:14]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x
    aux1, aux2 = u[15:16] #control variables
    g_ctrl_1, g_ctrl_2 = exp.([aux1, aux2]) #to ensure positive conductances
    g[ctrl_idx1] = g_ctrl_1
    g[ctrl_idx2] = g_ctrl_2
    @inbounds begin
        gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = g
    end

    du[1] = sum([(gNa) * mNa^3 * hNa^1 * (E_Na - v),
        (gCaS) * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
        (gCaT) * mCaT^3 * hCaT^1 * (DynECa(ca) - v),
        (gKa) * mKa^3 * hKa^1 * (E_K - v),
        (gKCa) * mKCa^4 * (E_K - v),
        (gKd) * mKd^4 * (E_K - v),
        (gH) * mH * (E_H - v),
        (gleak) * (E_L - v),
        (1) * inp[timeidx]
    ]) / Cm

    tmpICa = gCaS * mCaS^3 * hCaS * (DynECa(ca) - v) +
             gCaT * mCaT^3 * hCaT * (DynECa(ca) - v)

    du[2] = (1 / τCa) * (-ca + 0.05 + factorarea * tmpICa / Cm)
    du[3] = (1 / τav) * (-avg + ca)

    du[4:14] .= A_stg(v) * (B_stg(v, ca) - x)

    du[15] = γ * (1 / τ_g) * (((μ⁺ - ϵ1) - α * avg) / μ⁺) / exp(u[15])
    du[16] = γ * (1 / τ_g) * (((σ2⁺ - ϵ2) - β * avg) / σ2⁺) / exp(u[16])
    return nothing
end

function implicit_stats(neuronid, list_dataframes, target_mean)
    model_XY1 = linearmodel_XY1(list_dataframes, neuronid)
    model_XY2 = linearmodel_XY2(list_dataframes, neuronid)
    ϵ1, α = coef(model_XY1)
    ϵ2, β = coef(model_XY2)
    ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α #function that goes from mean to variance
    μ⁺ = target_mean
    σ2⁺ = ĥ₋(μ⁺, α, β, ϵ1, ϵ2) #variance target based on mean target
    return (target_mean, σ2⁺), (ϵ1, α), (ϵ2, β)
end

function control_with_g(neuronid, ctrl_idxs, init_cond, list_dataframes, input, τ_g, μ⁺, γ, tspan_long, save_time, dt)
    conductances = ActiveBank[neuronid, :]
    p = (Vector{Real}([conductances...]), input, dt, τ_g, implicit_stats(neuronid, list_dataframes, μ⁺)..., γ, ctrl_idxs)
    prob = ODEProblem(controller_ode!, init_cond, tspan_long, p)
    sol = OrdinaryDiffEq.solve(prob, TRBDF2(), dt=dt; saveat=save_time, maxiters=1e12,)
    return sol
end


τ_g = 20 * τav
neuronid = 136
conductances = ActiveBank[neuronid, :]
μ⁺ = 23
landmarks = implicit_stats(neuronid, list_stgvariance_dataframes, μ⁺)
(μ⁺, σ2⁺), (ϵ1, α), (ϵ2, β) = landmarks
avg_tgt = (μ⁺ - ϵ1) / α # = 2.5

γ = 0.5f0 # control strength, can be adjusted
T = 3 * τav
window = 14
step_up = [2, 3]

function outer_wrap(step_up, init_0, ctrl_idxs, window, T, conductances, path)

    ctrl_init = [log(conductances[ctrl_idxs[1]]), log(conductances[ctrl_idxs[2]])]
    init_cond = [init_0..., ctrl_init...]
    tspan_warm = (tspan[1], window * T)
    warm_start = control_with_g(neuronid, ctrl_idxs, Real.(init_cond), list_stgvariance_dataframes, vcat(pinknoise_input[(μᵢ, 1.0f0)][2:2+window-1]...), τ_g,
        μ⁺, γ, tspan_warm, tspan_warm[1]:dt:tspan_warm[2], dt)

    init_cond = warm_start[:, end]

    #############################################################
    # now control when step disturbance is applied
    # solver chunks each step and solutions are chained
    ###########################################################

    sol_full_u = []
    sol_full_t = []
    inp_full = []

    for (j, σᵢ) in enumerate(noise_levels[[step_up]...])

        tspan_long = (1 + (j - 1) * window * T, 1 + (j) * window * T)
        input = generate_pink_noise(length(tspan_long[1]:dt:tspan_long[2]); μ=0.0f0, σ2=σᵢ^2)

        sol_chunk = control_with_g(neuronid,  ctrl_idxs, init_cond, list_stgvariance_dataframes, input, τ_g,
            μ⁺, γ, tspan_long, tspan_long[1]:dt:tspan_long[2], dt)

        append!(sol_full_u, [sol_chunk.u])
        append!(sol_full_t, [sol_chunk.t])
        append!(inp_full, input)
        init_cond = sol_chunk[:, end]

    end
    sol_full_u = vcat(sol_full_u...)
    sol_full_t = vcat(sol_full_t...)

    sol_tot = DiffEqArray(sol_full_u, Float32.(sol_full_t))

    mean_vec, variance_vec = compute_running_ratestats(sol_tot, 12 * T)

    names = ["Na", "CaS", "CaT", "Ka", "KCa", "Kd", "H", "leak"]
    jldsave("$path/g$(names[ctrl_idxs[1]]),g$(names[ctrl_idxs[2]])_trajs.jld2"; sol=sol_tot, inp=inp_full, mean_vec = mean_vec, variance_vec = variance_vec)
    return sol_tot, inp_full

end

sol_tot, inp_full = outer_wrap(step_up, init_0, [7, 6], window, T, conductances, "data/raw/STG/control")