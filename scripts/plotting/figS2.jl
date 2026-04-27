"""
Supplemental figure on control 
"""

#################### compensation to a square pulse in input variance with g_syn 

neuronid = 136
μ⁺ = 40.0f0
γ = 0.1f0 # control strength, can be adjusted
Tburn = 18000
T = 18000
τ_g = 20 * τav
step_sq = [1, 2, 1]
#sol_w_gsyn, input, landmarks = control_generaldisturb_gsyn(step_sq, 14, neuronid, ActiveBank, list_stgvariance_dataframes, μ⁺, γ, τ_g, dt, Tburn);
#mean_vec_w_gsyn, variance_vec_w_gsyn = compute_running_ratestats(sol_w_gsyn, 12*T);
#jldsave("data/raw/STG/control/gsyn_sqrstep.jld2"; sol=sol_w_gsyn, input=input, landmarks=landmarks)

f = jldopen("data/raw/STG/control/gsyn_sqrstep.jld2", "r")
sol_w_gsyn = f["sol"]
landmarks = f["landmarks"]

plot_sqrstep = plot_step_syn_compensation(sol_w_gsyn, step_sq, landmarks, mean_vec_w_gsyn, variance_vec_w_gsyn, Tburn, T, [(5.7, 8.3), (5.5, 2.5)])
savefig("figures/sqr_step_gsyn_compensation.pdf")

################### compensation to step in input variance with g_cat

conductances = Vector(ActiveBank[neuronid, :])
μ⁺ = 20.0f0
σ2⁺, (ϵ1, α), (ϵ2, β) = implicit_stats(neuronid, list_stgvariance_dataframes, μ⁺)
target_acc = (μ⁺ - ϵ1) / α #target for average calcium
landmarks = (μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β))
T = 18000
step_up = [1,2]
windows=20
γ = 0.01

function control_gca_ode!(du, u, p, t)
    gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak = p[1]
    inp = p[2]
    dt = p[3]
    τ_g = p[4]
    μ⁺ = p[5]
    σ2⁺ = p[6]
    ϵ1, α = p[7]
    ϵ2, β = p[8]
    γ = p[9]

    v, ca, avg = u[1:3]
    x = u[4:14]
    mNa, hNa, mCaS, hCaS, mCaT, hCaT, mKa, hKa, mKCa, mKd, mH = x
    Δgcas, Δgcat = u[15:16] #control variables

    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    du[1] = sum([(gNa) * mNa^3 * hNa^1 * (E_Na - v),
        (gCaS + Δgcas) * mCaS^3 * hCaS^1 * (DynECa(ca) - v),
        (gCaT + Δgcat) * mCaT^3 * hCaT^1 * (DynECa(ca) - v),
        (gKa) * mKa^3 * hKa^1 * (E_K - v),
        (gKCa) * mKCa^4 * (E_K - v),
        (gKd) * mKd^4 * (E_K - v),
        (gH) * mH * (E_H - v),
        (gleak) * (E_L - v),
        (1) * inp[timeidx]
    ]) / Cm

    tmpICa =
        (gCaS + Δgcas) * mCaS^3 * hCaS * (DynECa(ca) - v) +
        (gCaT + Δgcat) * mCaT^3 * hCaT * (DynECa(ca) - v)

    du[2] = (1 / τCa) * (-ca + 0.05 + factorarea * tmpICa / Cm)
    du[3] = (1 / τav) * (-avg + ca)

    du[4:14] .= A_stg(v) * (B_stg(v, ca) - x)

    du[15] = γ * (1 / τ_g) * ((μ⁺ - ϵ1) - α * avg)
    du[16] = γ * (1 / τ_g) * ((σ2⁺ - ϵ2) - β * avg)
end

function control_with_gcat(conductances, input, init_cond, τ_g, μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_long, save_time, dt)
    p = (conductances, input, dt, τ_g, μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ)
    prob = ODEProblem(control_gca_ode!, init_cond, tspan_long, p)
    sol = OrdinaryDiffEq.solve(prob, TRBDF2(), dt=dt;
        saveat=save_time, maxiters=1e12,)
    return sol
end

function wrapper(step_up, windows, conductances, neuronid, list_active_dataframes, μ⁺, γ, τ_g, dt, T)
    #warm start to make sure neuron has reached target
    σ2⁺, (ϵ1, α), (ϵ2, β) = implicit_stats(neuronid, list_active_dataframes, μ⁺)

    sol_full_u = []
    sol_full_t = []
    inp_full = []

    init_cond = [u0..., 0.0, 0.0] # initial conditions including control variables
    tspan_warm = (tspan[1], windows*T)
    time_vec_warm = tspan_warm[1]:dt:tspan_warm[2]

    input_warm = generate_pink_noise(length(time_vec_warm); μ=0.0f0, σ2=noise_levels[2]^2)
    warm_start = control_with_gcat(conductances, input_warm, init_cond, τ_g,
        μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_warm, [tspan_warm[2]], dt)

    init_cond = [warm_start.u[1][1:end-2]...,0.0,0.0] #reset control variables to 0 after warmup
    conductances[2:3] .= conductances[2:3] .+ warm_start.u[1][15:16] #apply warmup control adjustments to base conductances

    #now give neuron multiple steps in input variance σᵢ∈[1,12]
    for (j,step) in enumerate(steps)
        @show j
        tspan_long = (1 + (j - 1) * windows * T, 1 + (j) * windows * T)
        input = generate_pink_noise(length(tspan_long[1]:dt:tspan_long[2]); μ=0.0f0, σ2=noise_levels[1+step]^2)
        sol_chunk = control_with_gcat(conductances, input, init_cond, τ_g,
            μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β), γ, tspan_long, tspan_long[1]:1.0:tspan_long[2], dt)

        append!(sol_full_u, [sol_chunk.u])
        append!(sol_full_t, [sol_chunk.t])
        append!(inp_full, input)
        init_cond = sol_chunk[:, end]
    end
    sol_full_u = vcat(sol_full_u...)
    sol_full_t = vcat(sol_full_t...)

    sol_tot = DiffEqArray(sol_full_u, Float32.(sol_full_t))
    return sol_tot, inp_full

end

#sol_full, inp_full = wrapper(steps, windows, conductances, neuronid, list_stgvariance_dataframes, μ⁺, γ, τ_g, dt, T);
#mean_vec_w_gcat, variance_vec_w_gcat = compute_running_ratestats(sol_w_gcat, 12*T);
# jldsave("data/raw/STG/control/gcat_upstep.jld2"; sol_full=sol_full, inp_full=inp_full)

f = jldopen("data/raw/STG/control/gcat_upstep.jld2", "r")
sol_w_gcat = f["sol_full"]
input = f["inp_full"]

function plot_step_gcat_compensation(sol_full, steps, landmarks, mean_vec, variance_vec, Tburn, T)

    plt_input = repeat([steps]..., inner=length(sol_full.t)÷length(steps))

    sol = sol_full[:, sol_full.t .>= Tburn]
    running_mean = mean_vec[sol_full.t .>= Tburn]
    running_variance = variance_vec[sol_full.t .>= Tburn]

    μ⁺, σ2⁺, (ϵ1, α), (ϵ2, β) = landmarks

    xticks_labeled = (3*T:T:length(sol.t), vcat(["T"], fill("", Int(sol.t[end] ÷ T) - 1), [""]))
    xticks_unlabeled = (2 * Tburn .+ Int(τ_g) .* [1, 2, 3, 4, 5], fill("", 5))
    xticks_longtimescale = (2*Tburn .+ Int(τ_g) .* [1, 2, 3, 4, 5], string.(1:5) .* "τᵤ")

    default(; lw=2.5, color=:black, xgrid=false, ygrid=false, labelfontsize=27, tickfontsize=17, legend=false,
        legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, yguidefontrotation=270)

    target_acc = (μ⁺ - ϵ1) / α #target for average calcium
    pcal = hline([target_acc], color=:grey, ls=:dash, ylabel=L"\langle[Ca]\rangle", lw=1.5)
    pcal = plot!(pcal, sol.t[1:10:end], sol[3, 1:10:end], xlabel="", lw=1.5,
        xticks=xticks_longtimescale, label="", yticks=[round(target_acc,digits=1)], )
    pcal = hline!([target_acc * 0.75, target_acc * 1.25], label="", lw=1, color=:grey, ls=:dash)

    pinp = plot(plt_input[1:end], ylabel=L"\sigma_{app}", xlabel="", label="",
        grid=false, xticks=xticks_unlabeled, yticks=[1, 2])

    p3 = plot(sol.t[1:10:end], (sol[15, 1:10:end] .+ conductances[2]) ./ conductances[2], label="",
        xticks=xticks_unlabeled, xlabel="", ylabel=L"g / g_0", color=:grey)
    p3 = plot!(sol.t[1:10:end], (sol[16, 1:10:end] .+ conductances[3]) ./ conductances[3], label="", yticks=[0.85,1] )
    p3 = annotate!(p3, 8 * T, 1.05, text(L"g_{CaT}", 27, :black))
    p3 = annotate!(p3, 15 * T, 1.05, text(L"g_{CaS}", 25, :grey))

    plt_err1 = hline([μ⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\mu",)
    plt_err1 = plot!(sol.t[1:10:end], running_mean[1:10:end], xticks=xticks_unlabeled,
         yticks=[10, 20],ylims=(7,21))

    plt_err2 = hline([σ2⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\sigma^2",)
    plt_err2 = plot!(sol.t[1:10:end], running_variance[1:10:end], 
        xticks=xticks_unlabeled, yticks=([50, 150, ], ["50", "150"]),ylims=(10,250) )

    my_layout = @layout [
        a{0.1h}
        b{0.245h}
        c{0.245h}
        d{0.245h}
        e{0.245h}
    ]

    plot(pinp, p3, plt_err1, plt_err2, pcal, layout=my_layout, legend=false,
        xlims = (2 * Tburn, sol.t[end]),
        size=(970,700), dpi=300, left_margin=16mm, bottom_margin=8mm, right_margin=1mm,)
end

plot_upstep = plot_step_gcat_compensation(sol_w_gcat, step_up, landmarks, mean_vec_w_gcat, variance_vec_w_gcat, Tburn, T)
savefig("figures/up_step_gcat_compensation.pdf")
