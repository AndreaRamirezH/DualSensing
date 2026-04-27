#Figure 3  collage

#individual (f,h) examples and voltage traces

function topleft_ylabel!(p, label; subplot=1) 
	xl = xlims(p[subplot])
 	yl = ylims(p[subplot]) 
# Place at ~2% from left, 95% up — in data coordinates
 	x_pos = xl[1] + 0.06 * (xl[2] - xl[1])
 	y_pos = yl[1] + 0.99 * (yl[2] - yl[1])
 	annotate!(p, x_pos, y_pos, text(label, :left, :top, 24), subplot=subplot)
end

# for ramp in variance
function sensingmodes_rampvar_normalized(list_of_dataframes, neuronid,  leg_place::Tuple)

    colorseries = palette(:viridis, 15)
    default(color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, lw=0.5, grid=false, legendfontsize=17, tickfontsize=15, labelfontsize=15)


    model_XY1 = linearmodel_XY1(list_of_dataframes, neuronid)
    X = model_XY1.mf.data.X
    Y1 = model_XY1.mf.data.Y1

    xmin, xmax = minimum(X), maximum(X)
    minlim = floor(xmin/xmax, digits=2)
    xticks = ([minlim * xmax, xmax], [string(minlim), "1"])

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    pl1 = plot(X, y1_pred, label="R²=$(round(r2_ofmean, digits=2))",  legend=leg_place[1], xticks=xticks)

    pl1 = scatter!(X, Y1,  ylabel="" , label="", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    ytick_positions, ytick_labels = Plots.yticks(pl1)[1]
    tick_positions = [ytick_positions[1], ytick_positions[end]]
    tick_labels = [ytick_labels[1], ytick_labels[end]]
    pl1 = plot!( yticks=(tick_positions, tick_labels))
    topleft_ylabel!(pl1,  L"\mu")

    ############################################################################

    model_XY2 = linearmodel_XY2(list_of_dataframes, neuronid)
    Y2 = model_XY2.mf.data.Y2
    #deleteat!(Y2, argmax(Y2)) #this line is used for -- class exclusively
    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    pl2 = plot(X, y2_pred, label="R²=$(round(r2_ofvar, digits=2))", legend=leg_place[2], yformatter=y -> y / 1_000, xticks=xticks)
    pl2 = scatter!(X, Y2, ylabel="", label="", color=[colorseries[i] for i in repeat(1:15, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    ytick_positions, ytick_labels = Plots.yticks(pl2)[1]
    tick_positions = [ytick_positions[1], ytick_positions[end-1], ytick_positions[end]]
    tick_labels = [ytick_labels[1], ytick_labels[end-1], "x10³"]
  
    pl2 = plot!(yticks=(tick_positions, tick_labels))
    topleft_ylabel!(pl2, L"\sigma^2")

    P = plot(pl1, pl2, layout=(1,2), size=(500,250), bottom_margin=1mm)

savefig(P, "../figures/STG/normalized$neuronid.svg")

end

sensingmodes_rampvar_normalized(list_stgvariance_dataframes, 59, (:bottomright, :bottomright))
sensingmodes_rampvar_normalized(list_stgvariance_dataframes, 41, (:bottomright, :topright))
sensingmodes_rampvar_normalized(list_stgvariance_dataframes, 335, (:bottomright, :topright))
sensingmodes_rampvar_normalized(list_stgvariance_dataframes, 302, (:bottomright, :topright))


function interpretvoltage_rampvar_STG(NeuronBank, neuronid, trial)

	default( legendfontsize=17, tickfontsize=15,)

    levels = [2, 6, 11]
    colorseries = palette(:viridis, 15)
    conductances = Tuple(NeuronBank[neuronid, :])

    xlims = (125000, 135000)

    subplts = Vector{Plots.Plot}(undef, length(levels))
    for (i, level) in enumerate(levels)
        matrix = load_object("data/raw/STG/inputs_rampvariance/variance_level$(level).jld2")
        downsample = 10
        inp_trace = plot(matrix[1:downsample:end, trial], label = "", ylabel= i == 1 ? L"I_{app} \; [nA] " : "", lw=0.3, color=:black, alpha=0.9, 
				xticks=false, xlims=xlims ./ 10, xaxis=false,
				yticks= i == 1 ? 2 : false, ylims=(-30, 30), yaxis=true, grid=false, guidefontsize=20)

        sol = single_trial(conductances, matrix[:, trial], tspan, time_sol)
        v_trace = Float32.(sol[1, :])
        avgca = Float32.(sol[3, end])

        tmp = plot(v_trace, label = "", ylabel= i == 1 ? L"v \; [mV] " : "", lw=1.2, color=colorseries[level], 
			xticks=false,  xlims=xlims, xaxis = false, 
			yticks=  i == 1 ? 2 : false, ylims=(-77, 45), yaxis=true,  grid=false, guidefontsize=22)
        avca = round(avgca, digits=1)

        subplts[i] = plot(tmp, inp_trace, layout=@layout([a{0.7h}; b]), subplot_spacing=0, link=:none)
    end

    plot(subplts..., layout=grid(1, length(levels)), size = (length(levels)*250, 250) , left_margin=10mm, bottom_margin=8mm)
    savefig("../figures/STG/v,i,$neuronid.pdf")

end

 interpretvoltage_rampvar_STG(ActiveBank, 59, 1)
 interpretvoltage_rampvar_STG(ActiveBank, 335, 1)
 interpretvoltage_rampvar_STG(ActiveBank, 41, 1)
 interpretvoltage_rampvar_STG(ActiveBank, 302, 1)

#now for InputDriven neurons

function interpretvoltageCS(NeuronBank, neuronid, trial)

    default( legendfontsize=17, tickfontsize=15,)

    levels = [2, 6, 11]
    colorseries = palette(:viridis, 15)
    conductances = Tuple(NeuronBank[neuronid, :])
    rhb = Float32(floor(rheobases[neuronid]))

    xlims = (125000, 135000)
    subplts = Vector{Plots.Plot}(undef, length(levels))
    for (i, level) in enumerate(levels)
        m = load("data/raw/InputDriven/inputs_rampvariance/rhbs$(rhb).jld2")
        matrix = m["noise_level_$(level)"]
        downsample = 10
        inp_trace = plot(matrix[1:downsample:end, trial], label = "", ylabel= i == 1 ? L"I_{app} \; [nA] " : "", lw=0.3, color=:black, alpha=0.9, 
				xticks=false, xlims=xlims ./ 10, xaxis=false,
				yticks= i == 1 ? 2 : false, ylims=(-30+rhb, 30+rhb), yaxis=true, grid=false, guidefontsize=20)

        sol = single_CS(conductances, matrix[:, trial], tspan, time_sol)
        v_trace = Float32.(sol[1, :])
        avgca = Float32.(sol[3, end])

        tmp = plot(v_trace, label = "", ylabel= i == 1 ? L"v \; [mV] " : "", lw=1.2, color=colorseries[level], 
			xticks=false,  xlims=xlims, xaxis = false, 
			yticks=  i == 1 ? 2 : false, ylims=(-77, 45), yaxis=true,  grid=false, guidefontsize=22)
        avca = round(avgca, digits=1)
        subplts[i] = plot(tmp, inp_trace, layout=@layout([a{0.7h}; b]), subplot_spacing=0, link=:none)
    end
    plot(subplts..., layout=grid(1, length(levels)), size = (length(levels)*250, 250) , left_margin=10mm, bottom_margin=8mm)
    savefig("../figures/InputDriven/CS_v,i,$neuronid.pdf")
end

interpretvoltageCS(SilentBank, 44, 1)






#Figure 4  collage


#for ramp in mean
function sensingmodes_rampmean_normalized(list_of_dataframes, neuronid, leg_place::Tuple)

    colorseries = palette(:batlow10)
    default(color=:black, legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, lw=0.5, grid=false, legendfontsize=17, tickfontsize=15, labelfontsize=15)
    Mean_levels = mean_levels[2:end]

    model_XY1 = linearmodel_XY1(list_of_dataframes, neuronid; levels=Mean_levels)
    X = model_XY1.mf.data.X
    Y1 = model_XY1.mf.data.Y1

    xmin, xmax = minimum(X), maximum(X)
    minlim = floor(xmin/xmax, digits=2)
    xticks = ([minlim * xmax, xmax], [string(minlim), "1"])

    y1_pred = predict(model_XY1)
    r2_ofmean = r2(model_XY1)
    pl1 = plot(X, y1_pred, label="R²=$(round(r2_ofmean, digits=2))",  legend=leg_place[1], xticks=xticks)

    pl1 = scatter!(X, Y1,  ylabel="" , label="", color=[colorseries[i] for i in repeat(1:10, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    ytick_positions, ytick_labels = Plots.yticks(pl1)[1]
    tick_positions = [ytick_positions[1], ytick_positions[end]]
    tick_labels = [ytick_labels[1], ytick_labels[end]]
    pl1 = plot!( yticks=(tick_positions, tick_labels))
    topleft_ylabel!(pl1,  L"\mu")

    ############################################################################

    model_XY2 = linearmodel_XY2(list_of_dataframes, neuronid; levels=Mean_levels)
    Y2 = model_XY2.mf.data.Y2
    #deleteat!(Y2, argmax(Y2)) #this line is used for -- class exclusively
    y2_pred = predict(model_XY2)
    r2_ofvar = r2(model_XY2)

    pl2 = plot(X, y2_pred, label="R²=$(round(r2_ofvar, digits=2))", legend=leg_place[2], yformatter=y -> y / 1_000, xticks=xticks)
    pl2 = scatter!(X, Y2, ylabel="", label="", color=[colorseries[i] for i in repeat(1:10, inner=15)], ms=4.5, mswidth=0, alpha=0.7)

    ytick_positions, ytick_labels = Plots.yticks(pl2)[1]
    tick_positions = [ytick_positions[1], ytick_positions[end-1], ytick_positions[end]]
    tick_labels = [ytick_labels[1], ytick_labels[end-1], "x10³"]
  
    pl2 = plot!(yticks=(tick_positions, tick_labels))
    topleft_ylabel!(pl2, L"\sigma^2")

    P = plot(pl1, pl2, layout=(1,2), size=(500,250), bottom_margin=1mm)

savefig(P, "../figures/STG/normalized$neuronid.svg")

end

sensingmodes_rampmean_normalized(list_stgmean_dataframes, 1071, (:bottomright, :bottomleft))
sensingmodes_rampmean_normalized(list_stgmean_dataframes, 322, (:bottomright, :bottomleft))
sensingmodes_rampmean_normalized(list_stgmean_dataframes, 674, (:bottomright, :bottomleft))
sensingmodes_rampmean_normalized(list_stgmean_dataframes, 488, (:bottomright, :bottomleft))

function interpretvoltage_rampmean_STG(NeuronBank, neuronid, trial)

	default( legendfontsize=17, tickfontsize=15,)

    levels = [1, 5, 9]
    colorseries = palette(:batlow10)
    conductances = Tuple(NeuronBank[neuronid, :])

    xlims = (125000, 135000)

    subplts = Vector{Plots.Plot}(undef, length(levels))
    for (i, level) in enumerate(levels)
        matrix = load_object("STG/data/simulations/inputs_rampmean/mean_level$(level).jld2")
        downsample = 10
        inp_trace = plot(matrix[1:downsample:end, trial], label = "", ylabel= i == 1 ? L"I_{app} \; [nA] " : "", lw=0.3, color=:black, alpha=0.9, 
				xticks=false, xlims=xlims ./ 10, xaxis=false,
				yticks= i == 1 ? 2 : false, ylims=(-5, 15), yaxis=true, grid=false, guidefontsize=20)

        sol = single_trial(conductances, matrix[:, trial], tspan, time_sol)
        v_trace = Float32.(sol[1, :])
        avgca = Float32.(sol[3, end])

        tmp = plot(v_trace, label = "", ylabel= i == 1 ? L"v \; [mV] " : "", lw=1.2, color=colorseries[level], 
			xticks=false,  xlims=xlims, xaxis = false, 
			yticks=  i == 1 ? 2 : false, ylims=(-77, 45), yaxis=true,  grid=false, guidefontsize=22)
        avca = round(avgca, digits=1)

        subplts[i] = plot(tmp, inp_trace, layout=@layout([a{0.7h}; b]), subplot_spacing=0, link=:none)
    end

    plot(subplts..., layout=grid(1, length(levels)), size = (length(levels)*250, 250) , left_margin=10mm, bottom_margin=8mm)
    savefig("../figures/STG/v,i,$neuronid.pdf")

end
interpretvoltage_rampmean_STG(ActiveBank, 1071, 1)
interpretvoltage_rampmean_STG(ActiveBank, 674, 1)
interpretvoltage_rampmean_STG(ActiveBank, 488, 3)
interpretvoltage_rampmean_STG(ActiveBank, 322, 3)
