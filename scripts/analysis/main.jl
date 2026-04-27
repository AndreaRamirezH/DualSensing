using Distributed
addprocs(30)

@everywhere begin

    using Pkg
    Pkg.activate(".")
    include("setup.jl")

    function rampmean_active(neuronid::Int64)
        fixed = 2.0f0 #variance
        for meanlevel in 1:length(mean_levels)
            inputs = pinknoise_input[(mean_levels[meanlevel], fixed)]
            readout_STG(ActiveBank, neuronid, inputs, meanlevel, "data/raw/STG/simulations/ramp_mean")
        end
    end

    function rampvariance_active(neuronid::Int64)
        fixed = 0.0f0 #mean
        for noiselevel in 1:length(noise_levels)
            inputs = pinknoise_input[(fixed, noise_levels[noiselevel])]
            readout_STG(ActiveBank, neuronid, inputs, noiselevel, "data/raw/STG/simulations/ramp_variance")
        end
    end

    function rampvariance_inputdriven(neuronid::Int64)
        try 
        if neuronid == 1 #save inputs as matrices with timeseries for every trial in columns
            
            for m in bases
                filename = "data/raw/InputDriven/inputs_rampvariance/rhbs$m.jld2"

                jldopen(filename, "w") do file   # open once in write mode
                    for i in eachindex(noise_levels)
                        mat = hcat(pinknoise_input[(m, noise_levels[i])]...)
                        file["noise_level_$i"] = mat # store each matrix with a unique key
                    end
                end
            end
            
        end
        catch e
        @warn "worker caught error $e"
        end
        
        locpath = "data/raw/InputDriven/simulations/ramp_variance"
        vmatrix = Matrix{Vector{Float32}}(undef, 5, length(noise_levels))
        rhb = Float32(floor(rheobases[neuronid]))
        for noiselevel in 1:length(noise_levels)
            tmp_row = readout_CS(SilentBank, neuronid, pinknoise_input, rhb, noiselevel, locpath)
            vmatrix[:, noiselevel] .= tmp_row
        end
        jldsave("$locpath/$neuronid.jld2"; vmatrix=vmatrix)
    end

end

@time pmap(neuronid -> rampvariance_inputdriven(neuronid), 1:size(SilentBank,1))