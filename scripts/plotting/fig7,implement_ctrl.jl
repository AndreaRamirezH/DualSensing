"""
Plot the results from implementation of control of rate statistics
Step compensation with either intrinsic or synaptic conductance
"""
τ_g = 20 * τav
neuronid=136
conductances = ActiveBank[neuronid, :]

############### intrinsic ###############
function plot_transient(warm_start, avg_tgt, (μ⁺, σ2⁺))
    #check that it converges to stable readouts
    mean_, variance_ = compute_running_ratestats(warm_start, 12 * T)

    pbot = plot(plot(exp.(warm_start[15, 1:10:end]), label="gNa"),
        plot(exp.(warm_start[16, 1:10:end]), label="gKd"),
        plot(mean_[1:10:end], label="mean") |> p -> hline!(p, [μ⁺], ls=:dash, label=""),
        plot(variance_[1:10:end], label="variance") |> p -> hline!(p, [σ2⁺], ls=:dash, label=""),
        layout=(2, 2))

    ptop = plot(warm_start[3, 1:10:end], label="<Ca>") |> p -> hline!([avg_tgt], label="")
    hline!([avg_tgt * 0.75, avg_tgt * 1.25], label="")

    plot(ptop, pbot, layout=(2, 1), size=(800, 400), legendfontsize=10,
        legendbackgroundcolor=:transparent, legendforegroundcolor=:transparent)
end

function plot_step_intr_compensation(sol_tot, ctrl_idxs, mean_vec, variance_vec, avg_tgt, steps, conductances)
    names = ["Na", "CaS", "CaT", "Ka", "KCa", "Kd", "H", "leak"]
    annot_fontsize = 21
    default(; marker=false, lw=2.5, grid=false, color=:black, 
        tickfontsize=17, legend=false, labelfontsize=27,
        legendbackgroundcolor=:transparent, legendforegroundcolor=:transparent, yguidefontrotation=270)
    
    xticks = (Int(τ_g) .* [1, 2, 3, 4, 5], string.(1:5) .* "τᵤ")
    xticks_unlabeled = (xticks[1], fill("",5))

    t0 = Int(τ_g / 5)
    pinput = plot(repeat([steps]..., inner=length(sol_tot.t) ÷ length(steps))[1:10:end],
        yticks = (steps, string.(Int.(noise_levels[steps]))), label="", ylabel=L"\sigma_{app}", xticks=xticks_unlabeled)

    pcal = plot(sol_tot[3, t0:10:end], label="", ylabel=L"\langle[Ca]\rangle", yticks=[round(avg_tgt,digits=1)], lw=1.5) |> p -> hline!([avg_tgt], label="", ls=:dash, color=:grey, lw=1.5)
    pcal = hline!([avg_tgt * 0.75, avg_tgt * 1.25], label="", lw=1, color=:grey, ls=:dash, xticks=xticks)


    pmid = plot(sol_tot.t[t0:10:end], exp.(sol_tot[15, t0:10:end]) ./ conductances[ctrl_idxs[1]], label="")
    pmid = plot!(sol_tot.t[t0:10:end], exp.(sol_tot[16, t0:10:end]) ./ conductances[ctrl_idxs[2]],label="", legend=:bottomright,
            ylabel=L"g/g_0", color=:grey, yticks=[0.75,1.25], xticks=xticks_unlabeled)
    pmid = annotate!(pmid, Int(2.4*τ_g), 1.4, text(L"g_{H}", 27))
    pmid = annotate!(pmid, Int(4*τ_g), 1.27, text(L"g_{Kd}", 22, :grey))


    plt_err1 = hline([μ⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\mu",)
    plt_err1 = plot!(sol_tot.t[t0:10:end], mean_vec[t0:10:end],label="",
        yticks=ceil.([μ⁺ * 0.75, μ⁺ * 1.25]), ylims=(μ⁺ * 0.5, μ⁺ * 1.5), xticks=xticks_unlabeled)

    
    plt_err2 = hline([σ2⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\sigma^2",)
    plt_err2 = plot!(sol_tot.t[t0:10:end], variance_vec[t0:10:end],label="",
        yticks=ceil.([σ2⁺ * 0.75, σ2⁺ * 1.25]), ylims=(σ2⁺ * 0.5, σ2⁺ * 1.5), xticks=xticks_unlabeled)


    my_layout = @layout [
        a{0.1h}
        b{0.245h}
        c{0.245h}
        d{0.245h}
        e{0.245h}
    ]

    plot(pinput, pmid, plt_err1, plt_err2, pcal,
        layout=my_layout,
        size=(970,700), dpi=300,
        left_margin=16mm, bottom_margin=8mm, right_margin=1mm)

end

f = jldopen("gH,gKd_trajs.jld2", "r")
sol_tot = f["sol_tot"]
inp_full = f["inp_tot"]
mean_vec = f["mean_vec"]
variance_vec = f["variance_vec"]
close(f)

avg_tgt = 2.5
step_up = [2, 3]

plot_step_intr_compensation(sol_tot, [7,6], mean_vec, variance_vec, avg_tgt, step_up, conductances)
savefig("figures/gHcompensation.pdf")

############### synaptic ###############

function plot_step_syn_compensation(sol_full, steps, landmarks, mean_vec, variance_vec, Tburn, T, positions)

    plt_input = repeat([steps]..., inner=length(sol_full.t)÷length(steps))

    positions_gsyn, positions_gint = positions
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
        xticks=xticks_longtimescale, label="", yticks=[round(target_acc,digits=1)], ylims=(2, 7))
    pcal = hline!([target_acc * 0.75, target_acc * 1.25], label="", lw=1, color=:grey, ls=:dash)

    pinp = plot(plt_input[1:end], ylabel=L"\sigma_{app}", xlabel="", label="",
        grid=false, xticks=xticks_unlabeled, yticks=[1, 2])

    p3 = plot(sol.t[1:10:end], sol[15, 1:10:end], label="",
        xticks=xticks_unlabeled, xlabel="", ylabel=L"g / g_0", color=:grey)
    p3 = plot!(sol.t[1:10:end], sol[16, 1:10:end], label="", yticks=([0, 6],["1", "6"]), ylims=(-1, 10))
    p3 = annotate!(p3, (positions_gsyn[1]) * T + Tburn, positions_gsyn[2], text(L"g_{syn}", 27, :black))
    p3 = annotate!(p3, (positions_gint[1]) * T + Tburn, positions_gint[2], text(L"g_{Na}", 22, :grey))

    plt_err1 = hline([μ⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\mu",)
    plt_err1 = plot!(sol.t[1:10:end], running_mean[1:10:end], xticks=xticks_unlabeled,
         yticks=[25, 50], ylims=(20, 55),)

    plt_err2 = hline([σ2⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\sigma^2",)
    plt_err2 = plot!(sol.t[1:10:end], running_variance[1:10:end], 
        xticks=xticks_unlabeled, yticks=([700, 1200, 1700], ["7", "12", "×10²"]), ylims=(500, 2100),)

    my_layout = @layout [
        a{0.1h}
        b{0.245h}
        c{0.245h}
        d{0.245h}
        e{0.245h}
    ]

    plot(pinp, p3, plt_err1, plt_err2, pcal, layout=my_layout, legend=false,
        xlims = (2 * Tburn, sol.t[end]),
        size=(970,700), dpi=300, left_margin=14mm, bottom_margin=8mm, right_margin=1mm,)
end

f = jldopen("gSyn,gNa_trajs.jld2", "r")
sol_full = f["sol"]
input = f["input"]
landmarks = f["landmarks"]
mean_vec = f["mean_vec"]
variance_vec = f["variance_vec"]
close(f)

step_up = [1,2]
Tburn = 18000
T = 18000
plot_step_syn_compensation(sol_full, step_up, landmarks, mean_vec, variance_vec, Tburn, T, [(4,9), (6, 3)])
savefig("figures/gsyncompensation.pdf")