include("load_format.jl")

using StatsPlots, StatsBase



function correlation_XY1(list_of_dataframes, neuronid, levels; monotonic=false)
    Df = []
    for nl in 1:length(levels)
        df = list_of_dataframes[nl]
        Df = vcat(Df, df[neuronid, :])
    end
    individual_neuron_data = DataFrame(Df)
    X = vcat(individual_neuron_data.avgCa...)
    Y1 = vcat(individual_neuron_data.meanR...)
    #Y1 = replace(Y1, Inf => 0.0)

    mask = .!isinf.(Y1)   # boolean mask
    X_sub = X[mask]
    Y1_sub = Y1[mask]

    res = monotonic ? corspearman(Float64.(X_sub), Float64.(Y1_sub)) : cor(Float64.(X_sub), Float64.(Y1_sub))
    return res
end

function correlation_XY2(list_of_dataframes, neuronid, levels; monotonic=false)
    Df = []
    for nl in 1:length(levels)
        df = list_of_dataframes[nl]
        Df = vcat(Df, df[neuronid, :])
    end
    individual_neuron_data = DataFrame(Df)
    X = vcat(individual_neuron_data.avgCa...)
    Y2 = vcat(individual_neuron_data.σ2ᵣ...)
    #Y2 = replace(Y2, NaN => 0.0)

    mask = .!isnan.(Y2)   # boolean mask: true if not NaN
    X_sub = X[mask]
    Y2_sub = Y2[mask]

    res = monotonic ? corspearman(Float64.(X_sub), Float64.(Y2_sub)) : cor(Float64.(X_sub), Float64.(Y2_sub))
    return res
end

function correlation_Ca_rstats(list_of_dataframes, path::String, levels) #saves to csv
    correlations1 = Float64[]
    for neuronid in 1:size(list_of_dataframes[1])[1]
        push!(correlations1, correlation_XY1(list_of_dataframes, neuronid, levels))
    end
    correlations2 = Float64[]
    for neuronid in 1:size(list_of_dataframes[1])[1]
        push!(correlations2, correlation_XY2(list_of_dataframes, neuronid, levels))
    end
    CSV.write("$path.csv", DataFrame(:μ=>correlations1,:σ=>correlations2))
end

function Spearmancorrelation_Ca_rstats(list_of_dataframes, path::String, levels)
    correlations1 = Float64[]
    for neuronid in 1:size(list_of_dataframes[1])[1]
        push!(correlations1, correlation_XY1(list_of_dataframes, neuronid, levels; monotonic=true))
    end
    correlations2 = Float64[]
    for neuronid in 1:size(list_of_dataframes[1])[1]
        push!(correlations2, correlation_XY2(list_of_dataframes, neuronid, levels; monotonic=true))
    end
    CSV.write("$path.csv", DataFrame(:μ => correlations1, :σ => correlations2))
end


###########################################################################
# Pearson correlation for active stg population with increasing input variance #

correlation_Ca_rstats(list_stgvariance_dataframes, "results/STG/corr_rampvariance", noise_levels)


# Spearman correlation identifies monotonic relationships, not necessarily linear #
#Spearmancorrelation_Ca_rstats(list_stgvariance_dataframes, "results/STG/SpearmanCorr_rampvariance", noise_levels)

###########################################################################
# Pearson correlation for active stg population with increasing input mean#

correlation_Ca_rstats(list_stgmean_dataframes, "results/STG/corr_rampmean", mean_levels)


###########################################################################
# Pearson correlation for input-driven ConnorStevens population with increasing input variance 
# all trials where MFR is 0 are excluded

correlation_Ca_rstats(list_csvariance_dataframes, "results/InputDriven/corr_rampvariance", noise_levels)

