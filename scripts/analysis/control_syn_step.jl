using Interpolations, JLD2, RecursiveArrayTools, DataFrames, GLM


function control_gsyn_ode!(du, u, p, t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = p[1]
    inp = p[2]
    dt = p[3]
    τ_g= p[4]
    μ⁺ = p[5]
    σ2⁺ = p[6]
    ϵ1, α = p[7]
    ϵ2, β = p[8]
    γ = p[9]
    reversed = p[10]

    v, ca, avg = u[1:3]
    x = u[4:14]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x
    Δg_int, gsyn = reversed ? reverse(u[15:16]) : u[15:16]

    du[1] = (
        (gNa + Δg_int) * mNa^3 * hNa * (E_Na - v) +
        gCaS * mCaS^3 * hCaS * (DynECa(ca) - v) +
        gCaT * mCaT^3 * hCaT * (DynECa(ca) - v) +
        gKa * mKa^3 * hKa * (E_K - v) +
        gKCa * mKCa^4 * (E_K - v) +
        gKd * mKd^4 * (E_K - v) +
        gH * mH * (E_H - v) +
        gleak * (E_L - v) +
        gsyn * inp(t)
    ) / Cm

    tmpICa = gCaS * mCaS^3 * hCaS * (DynECa(ca) - v) +
            gCaT * mCaT^3 * hCaT * (DynECa(ca) - v)

    du[2] = (1 / τCa) * (-ca + 0.05 + factorarea * tmpICa / Cm)
    du[3] = (1 / τav) * (-avg + ca)

    du[4:14] .= A_stg(v) * (B_stg(v, ca) - x)

    du[15] = γ * (1 / τ_g) * ((μ⁺ - ϵ1) - α * avg)  
    du[16] = γ * (1 / τ_g) * ((σ2⁺ - ϵ2) - β * avg) 
end

function control_with_gsyn(conductances, input, init_cond, τ_g, reversed, μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_long, save_time, dt)
    
    p = (conductances, input, dt, τ_g, μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, reversed)
    prob = ODEProblem(control_gsyn_ode!, init_cond, tspan_long, p)
    sol = OrdinaryDiffEq.solve(prob, TRBDF2(), dt=dt;
            saveat=save_time, maxiters=1e12,)
    return sol
end

function control_generaldisturb_gsyn(steps, windows, neuronid, Bank, list_active_dataframes, μ⁺, γ, τ_g, dt)

    conductances = Vector(Bank[neuronid, :])
    start_calcium = mean(list_active_dataframes[findfirst(x -> x == noise_levels[2], noise_levels)][neuronid, :avgCa])
    σ2⁺, (ϵ1, α), (ϵ2, β) = implicit_stats(neuronid, list_active_dataframes, μ⁺)
    reversed = false

    base = 1 #where input variance yields mean rate around target mean
    T = 18000
    sol_full_u = []
    sol_full_t = []
    inp_full = []

    ctrl_0 = reversed ? reverse([0.0f0, 1.0f0]) : [0.0f0, 1.0f0]
    init_cond = [u0[1], u0[2], start_calcium, u0[4:end]..., ctrl_0...] # initial conditions including control variables
    tspan_warm = (tspan[1],windows*T)
    input_warm = generate_pink_noise(length(0.0:dt:windows*T); μ=0.0f0, σ2=noise_levels[base+1]^2)
    N = length(input_warm)
    inp_times = (0:N-1) .* dt .+ tspan_warm[1]
    inp_vals = input_warm  # same length
    input_interp = LinearInterpolation(inp_times, inp_vals; extrapolation_bc=Flat())

    warm_start = control_with_gsyn(conductances, input_interp, init_cond, τ_g, reversed, 
                    μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_warm, [tspan_warm[2]], dt)
    
    init_cond = warm_start.u[1]
    
    for (j,step) in enumerate(steps)

        tspan_long = (1 + (j-1)*windows*T, 1 + (j) * windows* T)
        input = generate_pink_noise(length(tspan_long[1]:dt:tspan_long[2]); μ=0.0f0, σ2=noise_levels[base+step]^2)
        N = length(input)
        inp_times = (0:N-1) .* dt .+ tspan_long[1]
        inp_vals = input  # same length
        input_interp = LinearInterpolation(inp_times, inp_vals; extrapolation_bc=Flat())

        sol_chunk = control_with_gsyn(conductances, input_interp, init_cond, τ_g, reversed,
                    μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_long, tspan_long[1]:1.0:tspan_long[2], dt)

        append!(sol_full_u,[sol_chunk.u])
        append!(sol_full_t, [sol_chunk.t])
        append!(inp_full, input)
        init_cond = sol_chunk[:,end]

    end
    sol_full_u = vcat(sol_full_u...)
    sol_full_t = vcat(sol_full_t...)

    sol_tot = DiffEqArray(sol_full_u, Float32.(sol_full_t))

    landmarks = (μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β))
    return sol_tot, inp_full, landmarks
end

neuronid = 136
μ⁺ = 40.0f0
γ = 0.1f0 # control strength, can be adjuste
T = 18000
τ_g = 20 * τav

step_up = [1,2]
sol_full, input, landmarks = control_generaldisturb_gsyn(step_up, 17, neuronid, ActiveBank, list_stgvariance_dataframes, μ⁺, γ, τ_g, dt);
mean_vec, variance_vec = compute_running_ratestats(sol_full, 12*T);
jldsave("data/raw/STG/control/gSyn,gNa_trajs.jld2"; sol=sol_full, input=input, landmarks=landmarks, mean_vec = mean_vec, variance_vec = variance_vec)
