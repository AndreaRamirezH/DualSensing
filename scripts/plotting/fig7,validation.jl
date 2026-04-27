#Plot the comparison between full biophysical and simple approximate models

# get the data
f_full= jldopen("data/raw/STG/control/biophysical_gsyn_rampvariance.jld2","r")
sol_full= f_full["sol_full"]
p = f_full["p"]
running_stats_full = f_full["running_stats"]
RMSE_full = f_full["RMSEs"]
close(f_full)

f_simple = jldopen("data/raw/STG/control/simple_gsyn_rampvariance.jld2")
sol_simple= f_simple["sol_simple"]
predicted_acc = f_simple["predicted_acc"]
targets = f_simple["targets"]
σ2ramp = f_simple["input_conditions"]
close(f_simple)

# define functions

function plot_compare_models(targets, number_changes, σ2ramp, sol_full, sol_simple, predicted_acc, running_stats_full)
    μ⁺, σ2⁺, target_acc = targets
    Δgsyn_full = sol_full[15, :]
    Δgsyn_simple = sol_simple[1,:]

    default(;marker=false, lw=2.5, color=:black, xgrid=false, labelfontsize=27, tickfontsize=17, legend=false,
        legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, yguidefontrotation=270)
    downsample_step_duration = Int(floor((length(σ2ramp[1:10:end]) - 1) / number_changes))
    ctrlxticks_labeled = (1:downsample_step_duration:length(sol_simple.t[1:10:end]),vcat(["", "τᵤ", "2τᵤ", ],fill("",5), ["Time"]))
    ctrlxticks_unlabeled = (ctrlxticks_labeled[1], fill("", length(ctrlxticks_labeled[1])))

    x_formatted(x) = x / 1_000
    plt_cal = hline([target_acc], label="", color=:grey, ls=:dash, lw=1.5, ylabel=L"\langle[Ca]\rangle",)
    plt_cal = hline!([0.75*target_acc, 1.25*target_acc], label="", color=:grey, ls=:dash, lw=1)
    plt_cal = plot!(sol_simple.t[1:10:end], predicted_acc[1:10:end], color=:blue, label="",
                 yticks=[round(target_acc, digits=1)], xticks=ctrlxticks_labeled, xtickfontsize=19)
    plt_cal = plot!(sol_full.t[1:10:end], sol_full[3, 1:10:end], lw=1, color=:black, label="")

    plt_inp = plot(sol_simple.t[1:10:end], σ2ramp[1:10:end],
        ylabel=L"\sigma_{app}", xlabel="", label="", grid=false,
        xticks=ctrlxticks_unlabeled, xtickfontsize=10, yticks=[1,10])

    plt_u = plot((1 .+ Δgsyn_full[1:10:end]), color=:black, label="biophysical")
    plt_u = plot!((1 .+ Δgsyn_simple[1:10:end]) ./ 1, label="approximate",
        xlabel="", ylabel=L"g / g_0", color=:blue, yticks = [0.1,1.0], 
        xticks=ctrlxticks_unlabeled,legend=:topright, legendfontsize=16)#,legend_columns=2)

    plt_err1 = hline([μ⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\mu",)
    plt_err1 = plot!(sol_full.t[1:10:end], running_stats_full[1][1:10:end], # "RMSE = $(round(RMSEs[1]))", 
                    label="\n[Hz]", color=:black, xticks=ctrlxticks_unlabeled, 
                    yticks = ([0.5* μ⁺,  1.5*μ⁺], string.(Int.(round.([0.5* μ⁺,  1.5*μ⁺])))) ,
                    ylims=(0.1* μ⁺,  2*μ⁺))
    

    plt_err2 = hline([σ2⁺], color=:grey, ls=:dash, label="", lw=1.5, ylabel=L"\sigma^2",)
    plt_err2 = plot!(sol_full.t[1:10:end], running_stats_full[2][1:10:end], # "RMSE = $(round(RMSEs[2]))",
                    label="\n[Hz^2]", color=:black, xticks=ctrlxticks_unlabeled, 
                    yticks = ([0.5*σ2⁺, 1.5*σ2⁺], string.(Int.(round.([0.5*σ2⁺, 1.5*σ2⁺])))) ,
                    ylims=(0.1*σ2⁺, 3*σ2⁺))

    my_layout = @layout [
        a{0.1h}
        b{0.245h}
        c{0.245h}
        d{0.245h}
        e{0.245h}
    ]
    plot(plt_inp, plt_u, plt_err1, plt_err2, plt_cal, layout=my_layout, 
        xformatter=x_formatted, xlims=(1,ctrlxticks_unlabeled[1][10]),
        legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
        size=(673, 795), dpi=300,
        left_margin=14mm, bottom_margin=6mm, right_margin=1mm,top_margin=6mm,)
end

ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α #function that goes from mean to variance
function deform_implicit_relation(path)

    files = readdir("$path")
    scalings = 0.1:0.1:1.0

    p = plot(xlabel="μ(r)", ylabel="σ²(r)", legend=:topleft, size=(400, 400),
        grid=false, yformatter=y -> y / 1_000, legenforegroundcolor=:transparent, legendbackgroundcolor=:transparent)
    c = palette([:lightgrey, :black], length(scalings))
    for (i,file) in enumerate(files)
        avgCa, meanR, varR = load("$path/$(file)", "avgCa", "meanR", "varR")

        X = Float64.(avgCa)
        Y1 = Float64.(meanR[:, 1])
        Y2 = Float64.(varR[:, 1])

        data = DataFrame(X=X, Y1=Y1)
        model_XY1 = lm(@formula(Y1 ~ X), data)

        data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
        model_XY2 = lm(@formula(Y2 ~ X), data)

        ϵ1, α = coef(model_XY1)
        ϵ2, β = coef(model_XY2)

        meanr_range = Y1
        varr_range = ĥ₋.(meanr_range, α, β, ϵ1, ϵ2)
        p = plot!(meanr_range, varr_range, label="g_syn × $(scalings[i])", lw=2, color=c[i])
    end

    return p
end

function plot_linearmodel_gsyn(path)
    #plots the coefficients of the linear models f,h as gsyn (control variable) evolves

    y_alphas = []
    y_c1s = []
    y_betas = []
    y_c2s = []

    files = readdir("$path")[2:end]
    x = 0.2:0.1:1.0

    for (i,file) in enumerate(files)
        avgCa, meanR, varR = load("$path/$(file)", "avgCa", "meanR", "varR")

        X = Float64.(avgCa)
        Y1 = Float64.(meanR[:, 1])
        Y2 = Float64.(varR[:, 1])

        data = DataFrame(X=X, Y1=Y1)
        model_XY1 = lm(@formula(Y1 ~ X), data)

        data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
        model_XY2 = lm(@formula(Y2 ~ X), data)

        ϵ1, α = coef(model_XY1)
        ϵ2, β = coef(model_XY2)

        append!(y_alphas, α)
        append!(y_betas, β)
        append!(y_c1s, ϵ1)
        append!(y_c2s, ϵ2)
    end

    default(;label="", color=:black, marker=:o, labelfontsize=25, tickfontsize = 17)
    plts = [plot(x, y_alphas, ylabel=L"\alpha", yguidefontrotation=270, yticks=[8,10], xticks=(0.2:0.1:1.0, fill("",9))),
            plot(x, y_betas, ylabel=L"\beta", yguidefontrotation=270, yticks=[350,600], xticks=(0.2:0.1:1.0, fill("",9))),
            plot(x, y_c1s, ylabel=L"c_1", yguidefontrotation=270, yticks=[-2.5,5], xticks=(0.2:0.1:1.0, fill("",9))),
            plot(x, y_c2s, ylabel=L"c_2", yguidefontrotation=270, yticks=([-1200,-600, -400],["-12", "-6","×10²"]), 
            xlabel=L"g_{syn}", xlabelfontsize=27, xticks=(0.2:0.1:1.0, string.([0.2, fill("",7)..., 1.0]))), 
            ]
            
    plot(plts...,
    xgrid = false, 
    layout=(4,1), size=(600,800), left_margin=9mm, dpi=300)
end

function fit_x_2variables(folder, steps)
    files = filter(x -> occursin("neur", x), readdir("$folder"))
    X1 = Float64[]
    X2 = Float64[]
    Y = Float64[]
    for file in files
        avgCa, meanR, varR = load("$folder/$(file)", "avgCa", "meanR", "varR")

        append!(Y, Float64.(avgCa[1:15*(steps[end])]))
        append!(X1, fill(parse(Float64, split(file, "_gsyn")[2][1:3]), length(avgCa[1:15*(steps[end])]))) #gcat
        append!(X2, Float64.(repeat(noise_levels[1 .+ steps], inner=15))) #σ_app
    end

    # Example data
    df = DataFrame(X1=X1,
                    X2=X2,
                    Y = Y)

    # Fit affine model
    model = lm(@formula(Y ~ X1 * X2), df)
    return df, model
end


function plot_3d_fit(df, model; xticks=[0.1, 0.5, 1.0], yticks =[1,5,10])
    scatter(df.X1, df.X2, df.Y, xlabel=L"g_{syn}", ylabel=L"\sigma_{app}", zlabel=L"⟨[Ca]⟩",
        legend=false, ms=2.0, alpha= 0.7, color=:grey,)

    # Create a grid for the fitted surface
    x1_range = range(minimum(df.X1), maximum(df.X1), length=50)
    x2_range = range(minimum(df.X2), maximum(df.X2), length=50)
    z_pred = [predict(model, DataFrame(X1=x1, X2=x2))[1] for x1 in x1_range, x2 in x2_range]

    # Add fitted surface
    plot!(x1_range, x2_range, z_pred', st=:surface, alpha=0.5, xticks=xticks, yticks=yticks)
end


############################### PLOT ###############################
plot_compare_models(targets, 12, σ2ramp, sol_full, sol_simple, predicted_acc, running_stats_full)
savefig("figures/compare_full_simple.pdf")

#deform_implicit_relation("STG/data/modulation/136gsynvariance/")
plot_linearmodel_gsyn("data/raw/STG/control/deformations/")
savefig("figures/smooth_linearcoefs_gsyn.pdf")

df, model = fit_x_2variables("data/raw/STG/control/deformations", 1:10)
p3d = plot(plot_3d_fit(df, model), camera=(75,15), zlims=(1,5))

Δgsyn_simple = sol_simple[1,:]
domain_coordinates = DataFrame(X1=1.0 .+ Δgsyn_simple, X2=σ2ramp)
predicted_acc = predict(model, domain_coordinates)
p_cal = plot!(p3d, domain_coordinates.X1, domain_coordinates.X2, predicted_acc, color=:blue, lw=1.5, camera=(20,18))
savefig("figures/3dsimple_gsynvariance.pdf")