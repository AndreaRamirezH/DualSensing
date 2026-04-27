function plot_modulation_sharedax(neuronid)
    fileids = [12, 26, 28]
    leg_place = :bottomright
    colorseries = palette(:viridis, 15)
    y_formatted(y) = y / 1_000

    files = filter(x -> occursin("neur$(neuronid)", x), readdir("data/raw/STG/modulation/rand_combination"))

    mu_main = plot(xlims=(0,35), ylims=(0,180))
    var_main = plot(xlims=(0,35), ylims=(0,6.5e3))
    ###############################################
    for fi in fileids[1:1]

        file = filter(x -> occursin("_$(fi)_", x), files)[1]
        avgCa, meanR, varR = load("data/raw/STG/modulation/rand_combination/$(file)", "avgCa", "meanR", "varR")

        X = Float64.(avgCa)
        Y1 = Float64.(meanR[:, 1])
        Y2 = Float64.(varR[:, 1])

        data = DataFrame(X=X, Y1=Y1)
        model_XY1 = lm(@formula(Y1 ~ X), data)

        y1_pred = predict(model_XY1)
        r2_ofmean = r2(model_XY1)
        mu_main = plot!(mu_main, X, y1_pred, label="", lw=0.2, ls=:dash,
            color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
            legend=leg_place, grid=false,)

        mu_main = scatter!(mu_main, X, Y1, xlabel="", ylabel="", label="",
            color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)

        ############################################################################
        data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
        model_XY2 = lm(@formula(Y2 ~ X), data)

        y2_pred = predict(model_XY2)
        r2_ofvar = r2(model_XY2)

        var_main = plot!(var_main, X, y2_pred, label="", lw=0.2, ls=:dash, color=:black,
            legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, legend=leg_place,
            grid=false, yformatter=y -> y_formatted(y))

        var_main = scatter!(var_main, X, Y2, label="", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)
    end
    ##################################################

    plot!(mu_main, inset=bbox(0.2, 0.38, 0.5, 0.5), 
            frame=:box, 
            yticks=false,
            xticks=false, 
            subplot=2, 
            bg_inside=nothing)
    plot!(var_main, inset=bbox(0.2, 0.38, 0.5, 0.5), 
            frame=:box, 
            yticks=false,
            xticks=false, 
            subplot=2, 
            bg_inside=nothing)

    #first the original neuron
    model_XY1 = linearmodel_XY1(list_active_dataframes, neuronid)
    X = model_XY1.mf.data.X
    Y1 = model_XY1.mf.data.Y1

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    plot!(mu_main[2], X, y1_pred, label="", lw=0.2, ls=:dash,
        color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
        legend=leg_place, grid=false, xticks=false)

    scatter!(mu_main[2], X, Y1, xlabel="", ylabel="", label="",
        color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)

    ############################################################################
    model_XY2 = linearmodel_XY2(list_active_dataframes, neuronid)
    Y2 = model_XY2.mf.data.Y2
    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    y_formatted(y) = y / 1_000

    scatter!(var_main[2], X, Y2, grid=false, yformatter=y -> y_formatted(y), 
    label = "", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)

    #############################################################################

    file = filter(x -> occursin("_$(fileids[3])_", x), files)[1]
    avgCa, meanR, varR = load("STG/data/modulation/rand_combination/$(file)", "avgCa", "meanR", "varR")

    X = Float64.(avgCa)
    Y1 = Float64.(meanR[:, 1])
    Y2 = Float64.(varR[:, 1])

    data = DataFrame(X=X, Y1=Y1)
    model_XY1 = lm(@formula(Y1 ~ X), data)

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    plot!(mu_main[2], X, y1_pred, label="", lw=0.2, ls=:dash,
        color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
        legend=leg_place, grid=false, xticks=false,)

    scatter!(mu_main[2], X, Y1, xlabel="", ylabel="", label="",
        color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)

    ############################################################################
    data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
    model_XY2 = lm(@formula(Y2 ~ X), data)

    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    plot!(var_main[2], X, y2_pred, label="", lw=0.2, ls=:dash, color=:black,
        legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, legend=leg_place,
        grid=false, yformatter=y -> y_formatted(y))

    scatter!(var_main[2], X, Y2, label="", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)

    ###############################################################
    # palcement of main 2 subplots

    ytick_positions, ytick_labels = Plots.yticks(mu_main)[1]
    deleteat!(ytick_positions, [2, 4, 6])
    deleteat!(ytick_labels, [2, 4, 6])
    xtick_positions, xtick_labels = Plots.xticks(mu_main)[1]
    deleteat!(xtick_positions, [2, 4, 6])
    deleteat!(xtick_labels, [2, 4, 6])
    xtick_labels[end] = "μM"
    
    mu_main = plot!(mu_main[1], ytickfontsize=10, yticks=(ytick_positions, ytick_labels),
        xticks=(xtick_positions, xtick_labels), xtickfontsize=10,)
	topleft_ylabel!(mu_main,  L"\mu")

    xtick_positions, xtick_labels = Plots.xticks(var_main)[1]
    deleteat!(xtick_positions, [2, 4, 6])
    deleteat!(xtick_labels, [2, 4, 6])
    xtick_labels[end] = "μM"
    ytick_positions, ytick_labels = Plots.yticks(var_main)[1]
    deleteat!(ytick_positions, [2, 4, 6])
    deleteat!(ytick_labels, [2, 4, 6])
   
    ytick_labels[end] = "x10³"

    var_main = plot!(var_main[1], xticks=(xtick_positions, xtick_labels), xtickfontsize=10,
        yticks=(ytick_positions, ytick_labels), ytickfontsize=10)
    topleft_ylabel!(var_main,  L"\sigma^{2}")

    plot(mu_main, var_main, layout=(1, 2), size=(650, 300))
end
plot_modulation_sharedax(969)
savefig("../figures/STG/mod_sharedaxis_$neuronid.pdf")


# for modulation along a single conductance ray
function plot_summary_deformation(neuronid, whichconductance, scalings, scat)
    moducolorseries = palette([:lightgrey, :lightgrey], length(scalings))
    colorseries = palette(:viridis, 15)
    leg_place = :bottomright
    files = filter(x -> occursin("neur$(neuronid)_ray$(whichconductance)", x), readdir("data/raw/STG/modulation/along_ray"))
    files = files[scalings]
    
    #plot initialise
    mu = plot(legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
        legend=leg_place, grid=false,)

    y_formatted(y) = y / 1_000
    vari = plot(legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, legend=leg_place,
        grid=false, yformatter=y -> y_formatted(y))

    max_y1 = 0.0
    max_y2 = 0.0
    min_x = 30.0

    matrix_X=zeros(225)
    matrix_Y1=zeros(225)
    matrix_Y2=zeros(225)

    #plot modulated conductances
    for (i,file) in enumerate(files)
        avgCa, meanR, varR = load("data/raw/STG/modulation/along_ray/$(file)", "avgCa", "meanR", "varR")

        X = Float64.(avgCa)
        Y1 = Float64.(meanR[:, 1])
        Y2 = Float64.(varR[:, 1])

        min_x = minimum([min_x; X])
        max_y1 = maximum([max_y1; Y1])
        max_y2 = maximum([max_y2; Y2])

        data = DataFrame(X=X, Y1=Y1)
        model_XY1 = lm(@formula(Y1 ~ X), data)

        y1_pred = predict(model_XY1)
        r2_ofmean = r2(model_XY1)
        mu = plot!(mu, X, y1_pred, label="", lw=2.2,
            color=moducolorseries[i], legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent,
            legend=leg_place, grid=false,)
        if scat 
            mu = scatter!(X, Y1, xlabel="", ylabel="", label="",
            color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=2.5, mswidth=0.0, alpha=0.7)
        end

        data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
        model_XY2 = lm(@formula(Y2 ~ X), data)

        y2_pred = predict(model_XY2)
        r2_ofvar = r2(model_XY2)

        vari = plot!(vari, X, y2_pred, label="", lw=2.2, color=moducolorseries[i],
            legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, legend=leg_place,
            grid=false, yformatter=y -> y_formatted(y))
        if scat
            vari = scatter!(X, Y2, label="", color=[colorseries[i] for i in repeat(1:15, inner=15)],
            ms=2.5, mswidth=0.0, alpha=0.7)
        end
        
        matrix_X = hcat(matrix_X, X)
        matrix_Y1 = hcat(matrix_Y1, y1_pred)
        matrix_Y2 = hcat(matrix_Y2, y2_pred)

    end

    tick_positions, tick_labels = Plots.yticks(mu)[1]
    tick_positions = [tick_positions[1], tick_positions[end]]
    tick_labels = [tick_labels[1], tick_labels[end]]
    xtick_positions, xtick_labels = Plots.xticks(mu)[1]
    xtick_labels[end] = "μM"
    mu = plot!(mu, xticks=(xtick_positions, xtick_labels), xtickfontsize=10, 
	yticks=(tick_positions, tick_labels), ytickfontsize=10,)
	topleft_ylabel!(mu,  L"\mu")


    ytick_positions, ytick_labels = Plots.yticks(vari)[1]
    ytick_positions = [ytick_positions[1], ytick_positions[end-1], ytick_positions[end]]
    ytick_labels = [ytick_labels[1], ytick_labels[end-1], ytick_labels[end]]
    ytick_labels[end] = "x10³"

    xtick_positions, xtick_labels = Plots.xticks(vari)[1]
    xtick_labels[end] = "μM"
    vari = plot!(vari, xticks=(xtick_positions, xtick_labels), xtickfontsize=10,
        yticks=(ytick_positions, ytick_labels), ytickfontsize=10)
	topleft_ylabel!(vari,  L"\sigma^{2}")

    plot(mu, vari, layout=(1,2), size=(650,300), dpi=150, bottom_margin=3mm)
    return plot!()
end

plot_summary_deformation(969, 1, [2, 3, 4, 5], true)
savefig("../figures/STG/modulated969_gNa.pdf")
plot_summary_deformation(neuronid, 2, [4,5,6,7,8,9,10], true)
savefig("../figures/STG/modulated969_gCaS.pdf")

