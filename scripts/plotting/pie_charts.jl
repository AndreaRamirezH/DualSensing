"""
Pie charts to summarize classes in each population with two experimental protocols
1 ramp input variance
2 ramp input mean
"""

#population = "STG" 
population = "InputDriven"

function make_pie(population, experiment, symb)

    data_ = CSV.read("results/$population/$experiment.csv", DataFrame) #correlations 
    rename!(data_, names(data_) .=> [:x1, :x2])
    total_rows = size(data_, 1)

    # Define each class
    conditions = Dict(
    "{+ +}" => (row -> row.x1 > 0.5 && row.x2 > 0.5),
    "{+ —}" => (row -> row.x1 > 0.5 && row.x2 < -0.5),
    "{— +}" => (row -> row.x1 < -0.5 && row.x2 > 0.5),
    "{— —}" => (row -> row.x1 < -0.5 && row.x2 < -0.5),
    "{~₀ ~₀}" => (row -> (row.x1 > 0.5 && -0.5 < row.x2 < 0.5) ||
                            (-0.5 < row.x1 < 0.5 && row.x2 > 0.5) ||
                            (-0.5 < row.x1 < 0.5 && -0.5 < row.x2 < 0.5) ||
                            (-0.5 < row.x1 < 0.5 && row.x2 < -0.5) ||
                            (row.x1 < -0.5 && -0.5 < row.x2 < 0.5))
    )


    # Compute counts & percentages
    results = DataFrame(
        Condition = String[],
        Count = Int[],
        Percentage = Float64[]
    )

    for (label, condition) in conditions
        counted = count(condition, eachrow(data_))
        percentage = round(counted / total_rows * 100; digits=2)
        push!(results, (label, counted, percentage))
    end
    println(results)

    StatsPlots.pie(results.Condition,
        results.Percentage, title="Correlation classes as $(symb)_in increases",
        legend=:outerright, legendbackgroundcolor=:transparent, legendforegroundcolor=:transparent,
        size=(600, 600), color=[:lightblue, :lightgreen, :lightcoral, :lightgrey, :lightyellow],)
end

p1 = make_pie(population, "corr_rampvariance", "σ")
p2 = make_pie(population, "corr_rampmean", "μ")
plot(p1, p2, layout=(1, 2), size=(1200, 600), titlefontsize=16, legendfontsize=16,)
savefig("../figures/$population/pie_charts.pdf")


"""
Prob density function with marginals
"""
function marginalhistogram(filepath, contourlevels)
    raw_corrs = CSV.read("$filepath.csv", DataFrame)
    correlations1, correlations2 = raw_corrs.μ, raw_corrs.σ
    p = marginalkde(correlations1, correlations2, levels=contourlevels,  
        color=:black, lw=1.5,
        xlabel=L"\rho(\langle[Ca]\rangle,\mu(r))", ylabel=L"\rho(\langle[Ca]\rangle,\sigma^2(r))",
        tickfont=font("Arial"), xtickfontsize=16, ytickfontsize=16,
        legendforegroundcolor=:transparent, legendbackgroundcolor=:transparent, labelfontsize=17,
        size=(400, 400), dpi=1000,)
    p = Plots.plot!(xlims=(-1.2, 1.2), ylims=(-1.2, 1.2), ticks=[-1, 0, 1], subplot=2)
    p = Plots.plot!(xlims=(-1.2, 1.2), xticks=[-1, 0, 1], xtickfontsize=1, tickfontcolor=:white, subplot=1)
    p = Plots.plot!(ylims=(-1.2, 1.2), yticks=[-1, 0, 1], ytickfontsize=1, tickfontcolor=:white, subplot=3)

    for sp in (p.subplots[1], p.subplots[3])
        for series in sp.series_list
            series[:linewidth] = 2.5
        end
    end

    sp = p.subplots[2]
    s = sp.series_list[1]
    s[:seriestype] = :contourf
    s[:color] = :plasma
    x = s[:x]
    y = s[:y]
    z = s[:z].surf

    zmin, zmax = minimum(z), maximum(z)
    levels = range(zmin, zmax, length=contourlevels+2)  # +1 to get boundaries
    fill_levels = levels[2:end-1]  # fill in white first level

    empty!(sp.series_list)
    # Add filled contour
    Plots.plot!(sp, x, y, z,
        seriestype=:contourf,
        color=:YlOrBr,
        levels=collect(fill_levels),
        fillalpha=0.4,
    )
    Plots.plot!(sp, x, y, z,
        seriestype=:contour,
        levels= contourlevels,
        color=:black,
        lw=0.05
    )
    
    cbar = heatmap(rand(2, 2), clims=extrema(levels), framestyle=:none, c=cgrad(:YlOrBr, zmax),
        cbar=true, lims=(-1, 0), colorbar_title="", tickfontsize=12, titlefontsize=15)

    return p
end
marginalhistogram("results/STG/corr_rampvariance", 10)
savefig("../figures/STG/rampvariance_correlations_fillmarginalkde.pdf")

marginalhistogram("results/STG/corr_rampmean", 15)
savefig("../figures/STG/rampmean_correlations_fillmarginalkde.pdf")


