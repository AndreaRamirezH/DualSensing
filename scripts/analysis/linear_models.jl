using StatsPlots

include("load_format.jl")

function goodness_linear_fits(list_of_dataframes, filenamepath::String)

    goodnessXY1 = []
    coefXY1 = []
    for neuronid in 1:size(list_of_dataframes[1])[1]
        lm = linearmodel_XY1(list_of_dataframes, neuronid)
        push!(goodnessXY1, r2(lm))
        push!(coefXY1, coef(lm)[2])
    end

    goodness_XY2 = []
    coefXY2 = []
    for neuronid in 1:size(list_of_dataframes[1])[1]
        lm = linearmodel_XY2(list_of_dataframes, neuronid)
        push!(goodness_XY2, r2(lm))
        push!(coefXY2, coef(lm)[2])
    end

    CSV.write("$filenamepath.csv", DataFrame(:μ_R2 => goodnessXY1, :σ_R2 => goodness_XY2,
                                                :μ_coef => coefXY1, :σ_coef=>coefXY2))
end

function plot_goodness_linear_fits(filepath)
    df = CSV.read("$filepath.csv", DataFrame)
    goodnessXY1, goodnessXY2 = df.μ_R2, df.σ_R2
    
    boxplot([goodnessXY1, goodnessXY2],
        xticks=(1:2, ["μ(r) ~ a⟨[Ca]⟩+c_1", "σ²(r) ~ b⟨[Ca]⟩+c_2"]), ylabel="R²", legend=false,
        title="Goodness of linear fits", titlefontsize=11,
        color=:black, ms=2.5, mswidth=0.2, alpha=0.3, ylims=(-0.1, 1), grid=false, size=(800, 600))
end


###########################################################################
# For input-driven ConnorStevens population #
goodness_linear_fits(list_csvariance_dataframes, "results/InputDriven/scores")
plot_goodness_linear_fits("results/InputDriven/scores")

###########################################################################
# For active stg population #
goodness_linear_fits(list_stgvariance_dataframes, "results/STG/scores")
plot_goodness_linear_fits("results/STG/scores")

