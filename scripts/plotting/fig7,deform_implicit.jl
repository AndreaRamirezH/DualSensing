"""
The implicit relationship hat{h} is deformed as conductances are modulated 
"""

y_formatted(y) = y / 1_000

function deform_implicit_relation_gcat(neuronid, which_scaling,)
    files = filter(x -> occursin("neur$(neuronid)_ray3", x), readdir("data/raw/STG/modulation/g_cat"))
    files = files[which_scaling]

    scalings = [split(file, "_ray3_")[2] |> f -> split(f, ".jld2")[1] for file in files]

    p = plot(xlabel=L"\mu(r) ", ylabel="", legend=:topleft, size=(650, 400),
        grid=false, yformatter=y -> y_formatted(y), legenforegroundcolor=nothing, legendbackgroundcolor=nothing)
    p = vline!([40], label="", color=:black, ls=:dash, lw=0.5)
    c = cgrad(:bamako10)[1:2:9]

    for (i,file) in enumerate(files)
        avgCa, meanR, varR = load("data/raw/STG/modulation/g_cat/$(file)", "avgCa", "meanR", "varR")

        X = Float64.(avgCa)
        Y1 = Float64.(meanR[:, 1])
        Y2 = Float64.(varR[:, 1])

        data = DataFrame(X=X, Y1=Y1)
        model_XY1 = lm(@formula(Y1 ~ X), data)

        data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
        model_XY2 = lm(@formula(Y2 ~ X), data)

        ϵ1, α = coef(model_XY1)
        ϵ2, β = coef(model_XY2)
        ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α

        meanr_range = Y1
        varr_range = ĥ₋.(meanr_range, α, β, ϵ1, ϵ2)
        p = plot!(meanr_range, varr_range, label="gCaT × $(scalings[i])", lw=3, color=c[i])
        scatter!([40], [ĥ₋(40, α, β, ϵ1, ϵ2)], label="", color=:grey, ms=7, alpha=0.5, markerstrokewidth=0) #target point
    end

    p = plot!(xlims=(5, 72), ylims=(0, 2500))

    ytick_positions, ytick_labels = Plots.yticks(p)[1]
    ytick_labels[end] = "x10³"
    ytick_labels[[2,4]] .= ""
    p = plot!(tickfontsize=17, yticks=(ytick_positions, ytick_labels), xticks=[10, 40, 70],
        labelfontsize=24, legendfontsize=15, fg_legend=:false, size=(1.52 * 300, 1.18 * 300),
        left_margin= 2mm, bottom_margin=2mm)
    return p
end
p3 = deform_implicit_relation_gcat(136, 1:2:9)

function deform_implicit_relation_gsyn(neuronid)
   
    files = filter(x -> occursin("neur$(neuronid)_", x), readdir("data/raw/STG/modulation/g_syn"))

    p = plot(xlabel=L"\mu(r)", ylabel="", legend=:topleft, size=(650, 400),
        grid=false, yformatter=y -> y_formatted(y), legenforegroundcolor=nothing, legendbackgroundcolor=nothing)
    p = vline!([40], label="", color=:black, ls=:dash, lw=0.5)

    c = cgrad(:bamako10)[1:2:9]
    for (i, file) in enumerate(files[1:5])
        avgCa, meanR, varR = load("data/raw/STG/modulation/g_syn/$(file)", "avgCa", "meanR", "varR")

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
        ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α
        varr_range = ĥ₋.(meanr_range, α, β, ϵ1, ϵ2)
        p = plot!(meanr_range, varr_range, label="gsyn × $(i)", lw=3, color=c[i])
        scatter!([40], [ĥ₋(40, α, β, ϵ1, ϵ2)], label="", color=:grey, ms=7, alpha=0.5, markerstrokewidth=0) #target point
    end

    p = plot!(xlims=(5, 72), ylims=(0, 2500))

    ytick_positions, ytick_labels = Plots.yticks(p)[1]
    ytick_labels[end] = "x10³"
    ytick_labels[[2, 4]] .= ""
    p = plot!(tickfontsize=17, xticks=[10,40,70],
        yticks=(ytick_positions, ytick_labels),
        labelfontsize=24, legendfontsize=15, fg_legend=:false, size=(1.52 * 300, 1.18 * 300),
        left_margin=5mm, bottom_margin=5mm)
    return p

end
p2 = deform_implicit_relation_gsyn(136)

function deform_implicit_relation_gH(neuronid)

    files = filter(x -> occursin("neur$(neuronid)_ray7", x), readdir("data/raw/STG/modulation/g_h"))
    files = files[3:end]

    p = plot(xlabel=L"\mu(r)", ylabel=L"\sigma^2(r)", legend=:topleft, size=(650, 400),
        grid=false, yformatter=y -> y_formatted(y), legenforegroundcolor=nothing, legendbackgroundcolor=nothing)
    p = vline!([40], label="", color=:black, ls=:dash, lw=0.5)

    c = cgrad(:bamako10)[1:2:9]
    scalings = [split(file, "_ray7_")[2] |> f -> split(f, ".jld2")[1] for file in files]
    for (i, file) in enumerate(files[1:end])
        avgCa, meanR, varR = load("data/raw/STG/modulation/g_h/$(file)", "avgCa", "meanR", "varR")

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
        ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α
        varr_range = ĥ₋.(meanr_range, α, β, ϵ1, ϵ2)
        p = plot!(meanr_range, varr_range, label="gH × $(scalings[i])", lw=3, color=c[i])
        scatter!([40], [ĥ₋(40, α, β, ϵ1, ϵ2)], label="", color=:grey, ms=7, alpha=0.5, markerstrokewidth=0) #target point
    end

    p = plot!(xlims=(5, 72), ylims=(0, 2500))

    p = plot!(tickfontsize=17, xticks=[10, 40, 70],
        yticks=([0:500:2500...], ["0.0", "", "1.0", "", "2.0", "x10³"]),
        labelfontsize=24, legendfontsize=15, fg_legend=:false, size=(1.52 * 300, 1.18 * 300),
        left_margin=5mm, bottom_margin=5mm)
    return p

end
p1 = deform_implicit_relation_gH(136)

plot(p1,p2,p3,
     layout=(1, 3), size=(1500,450), dpi=300, 
     left_margin=10Plots.mm, bottom_margin=10Plots.mm)
savefig("../figures/STG/deform_implicit.pdf")
