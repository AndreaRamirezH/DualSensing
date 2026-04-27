using Plots, Plots.Measures, Statistics, ColorSchemes, LaTeXStrings, GLM

function load_dataframes(folder)
    files = filter(f -> endswith(f, ".jld2"), readdir(folder))
    sorted_files = sort(files, by = x -> parse(Int, split(x, "neur_trials")[1]))

    Df = DataFrame(
        :id => Int64[],
        :σ2ᵥ => Vector[],
        :σ2ᵣ => Vector[],
        :B => Vector[],
        :avgCa => Vector[],
        :meanV => Vector[],
        :meanR => Vector[],)

    for (id,f) in enumerate(sorted_files)
        avgCa,meanV,varV,spktms,pattern = load("$folder/$f", 
                            "rowtrial_avgCa",
                            "rowtrial_meanV",
                            "rowtrial_varV",
                            "rowtrial_spktms",
                            "rowtrial_pattern")
                            
        varR = var.(insta_rate.(spktms)) #variance of instantaneous firing rate
        meanR = mean.(insta_rate.(spktms)) #firing rate r = <1/ISI> for each trial

        df = DataFrame(:id=> id,
                       :σ2ᵥ=> [vec(varV)],
                       :σ2ᵣ => [vec(varR)],
                       :B=> [vec(pattern)],
                       :avgCa=> [vec(avgCa)],
                       :meanV=> [vec(meanV)],
                       :meanR => [vec(meanR)],)
        Df = vcat(Df, df)
    end
    return Df
end

list_csvariance_dataframes = [load_dataframes("data/raw/InputDriven/simulations/ramp_variance/noiselevel$(x)") for x in 1:length(noise_levels)]
for (i,el) in enumerate(list_csvariance_dataframes)
    save("data/processed/InputDriven/ramp_variance/stats$i.jld2", "df", el)
end

list_stgvariance_dataframes = [load_dataframes("data/raw/STG/simulations/ramp_variance/noiselevel$(x)") for x in 1:length(noise_levels)]
for (i,el) in enumerate(list_stgvariance_dataframes)
    save("data/processed/STG/ramp_variance/stats$i.jld2", "df", el)
end
list_stgmean_dataframes = [load_dataframes("data/raw/STG/simulations/ramp_mean/mean$(x)") for x in 1:10]
for (i,el) in enumerate(list_stgmean_dataframes)
    save("data/processed/STG/ramp_mean/stats$i.jld2", "df", el)
end