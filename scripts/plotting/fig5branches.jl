function two_branches(list_stgvariance_dataframes, list_stgmean_dataframes, neuronid)

     default(color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, lw=0.5, grid=false, legendfontsize=17, tickfontsize=12, labelfontsize=15)

    colorseries = palette(:viridis, 15)

    model_XY1 = linearmodel_XY1(list_stgvariance_dataframes, neuronid)
    X = model_XY1.mf.data.X
    x_min_val, x_max_val = extrema(X)
    Y1 = model_XY1.mf.data.Y1
    y1_max_val = maximum(Y1)

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    pl1 = plot(X, y1_pred, label="", lw=0.5, xticks=false)

    pl1 = scatter!(pl1, X, Y1, xlabel="", ylabel="", label="",
        color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    ###############################################################
    model_XY2 = linearmodel_XY2(list_stgvariance_dataframes, neuronid)
    Y2 = model_XY2.mf.data.Y2
    y2_max_val = maximum(Y2)

    #deleteat!(Y2, argmax(Y2)) #this line is used for -- class exclusively
    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    y_formatted(y) = y / 1_000

    pl2 = plot(X, y2_pred, label="", lw=0.5, yformatter=y -> y_formatted(y))

    pl2 = scatter!(pl2, X, Y2, label="", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=4.5, mswidth=0, alpha=0.7)
    
    ######## now the other branch ########

    colorseries = palette(:batlow10)

    model_XY1 = linearmodel_XY1(list_stgmean_dataframes, neuronid; levels=Mean_levels)
    X = model_XY1.mf.data.X
    Y1 = model_XY1.mf.data.Y1

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    pl1 = plot!(pl1, X, y1_pred, label="", lw=0.5, color=:black, grid=false, xticks=false)

    pl1 = scatter!(pl1, X, Y1, xlabel="", ylabel="", label="",
        color=[colorseries[i] for i in repeat(1:10, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    tick_positions, tick_labels = Plots.yticks(pl1)[1]
    tick_positions = [tick_positions[1], tick_positions[length(tick_positions)÷2+1], tick_positions[end]]
    tick_labels = [tick_labels[1], tick_labels[length(tick_positions)÷2+1], tick_labels[end]]
    pl1 = plot!(yticks=(tick_positions, tick_labels))

    topleft_ylabel!(pl1,  L"\mu")

    ############################################################################
    model_XY2 = linearmodel_XY2(list_stgmean_dataframes, neuronid; levels=Mean_levels)
    Y2 = model_XY2.mf.data.Y2
    #deleteat!(Y2, argmax(Y2)) #this line is used for -- class exclusively
    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    pl2 = plot!(pl2, X, y2_pred, label="", lw=0.5,  yformatter=y -> y_formatted(y))

    pl2 = scatter!(pl2, X, Y2, label="", color=[colorseries[i] for i in repeat(1:10, inner=15)], ms=4.5, mswidth=0, alpha=0.7)
    
    ytick_positions, ytick_labels = Plots.yticks(pl1)[1]
    ytick_labels[2] = ""

    ytick_positions, ytick_labels = Plots.yticks(pl2)[1]
    ytick_labels[2] = ""
    ytick_labels[end] = "x10³"

    xmin, xmax = minimum([x_min_val,minimum(X)]), maximum([x_max_val,maximum(X)])
    minlim = floor(xmin / xmax, digits=2)
    xticks = ([minlim * xmax, xmax], [string(minlim), "1"])

    pl2 = plot!(xticks=xticks,  yticks=(ytick_positions, ytick_labels),)
   topleft_ylabel!(pl2, L"\sigma^2")

    plot(pl1, pl2, layout=@layout([a{0.5h}; b]), size=(400, 400))
    savefig("../figures/STG/branches_normalized$neuronid.pdf")
end
two_branches(list_stgvariance_dataframes, list_stgmean_dataframes, 22)

function interpretbranch_var(NeuronBank, neuronid, trial)

	default( legendfontsize=17, tickfontsize=15,)

    levels = [2, 6, 11]
    colorseries = palette(:viridis, 15)
    conductances = Tuple(NeuronBank[neuronid, :])

    xlims = (120000, 135000)

    subplts = Vector{Plots.Plot}(undef, length(levels))
    for (i, level) in enumerate(levels)
        matrix = load_object("STG/data/simulations/inputs_rampvariance/variance_level$(level).jld2")
        downsample = 10
        inp_trace = plot(matrix[1:downsample:end, trial], label = "", ylabel= L"I_{app} \; [nA] " , lw=0.3, color=:black, alpha=0.9, 
				xticks=false, xlims=xlims ./ 10, xaxis=false,
				yticks= 2 , ylims=(-30, 30), yaxis=true, grid=false, guidefontsize=17)

        sol = single_trial(conductances, matrix[:, trial], tspan, time_sol)
        v_trace = Float32.(sol[1, :])
        avgca = Float32.(sol[3, end])

        tmp = plot(v_trace, label = "", ylabel=  L"v \; [mV] " , lw=1.2, color=colorseries[level], 
			xticks=false,  xlims=xlims, xaxis = false, 
			yticks=  2, ylims=(-77, 45), yaxis=true,  grid=false, guidefontsize=20)
        avca = round(avgca, digits=1)

        subplts[i] = plot(tmp, inp_trace, layout=@layout([a{0.7h}; b]), subplot_spacing=0, link=:none)
    end

    plot(subplts..., layout=grid(length(levels),1), size = (400, 600) , left_margin=10mm, bottom_margin=8mm)
    savefig("../figures/STG/var_v,i,$neuronid.pdf")

end
interpretbranch_var(ActiveBank, 22, 1)

function interpretbranch_mean(NeuronBank, neuronid, trial)

	default( legendfontsize=17, tickfontsize=15,)

    levels = [1, 5, 9]
    colorseries = palette(:batlow10)
    conductances = Tuple(NeuronBank[neuronid, :])

    xlims = (120000, 135000)

    subplts = Vector{Plots.Plot}(undef, length(levels))
    for (i, level) in enumerate(levels)
        matrix = load_object("STG/data/simulations/inputs_rampmean/mean_level$(level).jld2")
        downsample = 10
        inp_trace = plot(matrix[1:downsample:end, trial], label = "", ylabel= " " , lw=0.3, color=:black, alpha=0.9, 
				xticks=false, xlims=xlims ./ 10, xaxis=false,
				yticks= 2 , ylims=(-5, 15), yaxis=true, grid=false, guidefontsize=15)

        sol = single_trial(conductances, matrix[:, trial], tspan, time_sol)
        v_trace = Float32.(sol[1, :])
        avgca = Float32.(sol[3, end])

        tmp = plot(v_trace, label = "", ylabel=  " " , lw=1.2, color=colorseries[level], 
			xticks=false,  xlims=xlims, xaxis = false, 
			yticks=  2, ylims=(-77, 45), yaxis=true,  grid=false, guidefontsize=15)
        avca = round(avgca, digits=1)

        subplts[i] = plot(tmp, inp_trace, layout=@layout([a{0.7h}; b]), subplot_spacing=0, link=:none)
    end

    plot(subplts..., layout=grid(length(levels),1), size = (400, 600) , left_margin=10mm, bottom_margin=8mm)
    savefig("../figures/STG/mean_v,i,$neuronid.pdf")

end