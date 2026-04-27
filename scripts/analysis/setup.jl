
include("src/RateStats.jl")
using Main.RateStats  

using OrdinaryDiffEq, LinearAlgebra, Random, Statistics, Symbolics, LaTeXStrings
using CSV, DataFrames, JLD2, StatsBase, Plots, Plots.Measures, ColorSchemes

BursterBank = CSV.read("data/raw/STG/Burstingconductances.csv", DataFrame)
TonicBank = CSV.read("data/raw/STG/Tonicconductances.csv", DataFrame)
ActiveBank = vcat(BursterBank, TonicBank)

SilentBank = CSV.read("data/raw/InputDriven/CSconductances.csv", DataFrame)


"""
Now recover stats
"""

Mean_levels = mean_levels[2:end]

list_stgvariance_dataframes = [load("data/processed/STG/ramp_variance/stats$x.jld2", "df") for x in 1:length(noise_levels)]
list_stgmean_dataframes = [load("data/processed/STG/ramp_mean/stats$x.jld2", "df") for x in 1:length(Mean_levels)]

list_csvariance_dataframes = [load("data/processed/InputDriven/ramp_variance/stats$x.jld2", "df") for x in 1:length(noise_levels)]


#Keep in dataframes only those trials that don't go into depolarization block: filter by mean voltage above a cutoff


function filter_row(row, i::Symbol, constant)
    mask = row[i] .< constant
    return [col[mask] for col in row]
end

function filter_depolarizationblock(df)

    filtered = [filter_row(row, :meanV, -45.0) for row in eachrow(df[:, 2:end])]
    filtered_df = DataFrame([name => [row[j] for row in filtered] for (j, name) in enumerate(names(df[:, 2:end]))])
    # Add first column (:ID)
    filtered_df = insertcols!(filtered_df, 1, :id => df[:, 1])
    return filtered_df
end

filteredlist = filter_depolarizationblock.(list_csvariance_dataframes)
discard = unique(vcat([findall(id -> unique(df[id, :meanR]) == [], 1:size(df, 1)) for df in filteredlist]...))
keepids = setdiff(1:size(SilentBank, 1), discard)

list_csvariance_dataframes = [df[keepids, :] for df in filteredlist]



