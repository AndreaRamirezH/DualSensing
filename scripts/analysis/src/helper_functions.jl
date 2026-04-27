########### burstiness of voltage ##########
function find_spiketimes(v::Vector; detect=0, time_sol = time_sol)
    a = Int[]
    for i = 2:length(v)
        if v[i] > detect && v[i-1] < detect
            append!(a, i)
        end
    end
    time_indices = round.(time_sol[a], digits=2)
    return time_indices
end

function isi(a::Array{Float32,1})
    if length(a) < 2
        return [0.0f0]
    else
        isi = zeros(eltype(a),length(a) - 1)
        @inbounds for i = 1:length(a)-1
            isi[i] = a[i+1] - a[i]
        end
        return isi
    end
end

function SPB_PER_DC_IBFfunc(ISI::Array{Float32,1})
    # Computes spike per burst (SPB), period (PER), duty cycle (DC) and mean intraburst frequency (IBF)
    minISI = minimum(ISI)
    maxISI = maximum(ISI)
    interburst = findall(x -> x > maxISI / 3, ISI)
    intraburst = findall(x -> x < maxISI / 3, ISI)
    if length(intraburst) > 0
        SPB = round(length(intraburst) / length(interburst))
        IBP = mean(ISI[intraburst])  #Intra-burst period
        Burstdur = IBP * SPB
        PER = Burstdur + mean(ISI[interburst])
        DC = Burstdur / PER
        IBF = 1000 / IBP
        return (Float32(SPB), Float32(PER), Float32(DC), Float32(IBF))
    else
        return (0.0f0, 1.0f0, 0.0f0, 0.0f0)
    end
end


function bursts(spiketimes::Vector)
    Spb, per, dc, Ibf = SPB_PER_DC_IBFfunc(isi(spiketimes))
    burstiness = Spb * Ibf / per
end

function Burstiness(spiketimes::Vector)
    B = try bursts(spiketimes)
    catch err
        NaN
    end
    return B
end

########### firing rate of voltage ##########
nanmean(x) = mean(filter(!isnan, x))
nanmean(x, y) = mapslices(nanmean, x; dims= y)

nanvar(x) = var(filter(!isnan, x))
nanvar(x, y) = mapslices(nanvar, x; dims=y)

function insta_rate(spike_times::Vector)
    ISI_ms = isi(spike_times)
    ISI_s = ISI_ms ./ 1000
    inst_rate = 1 ./ ISI_s
end

function counting_rate(spike_times::Vector; ti = Ti, tf=Tf)
    duration_s = (tf - ti) / 1000  # ms → s
    num_spikes = length(spike_times)
    count_rate = num_spikes / duration_s
end

function compute_running_ratestats(sol, T) #need burn in period of T 

    spktms = find_spiketimes(sol[1, :]; time_sol=sol.t)
    n = length(spktms)

    inst_rates = insta_rate(spktms)
    rate_times = spktms[1:end-1]

    # For each t in sol.t, compute mean of inst_rates in window [t-T, t]
    mean_rates = zeros(length(sol.t))
    variance_rates = zeros(length(sol.t))

    for (j, t) in enumerate(sol.t)
        # Define the time window
        window_start = t - T

        # Find instantaneous rates within this window: rate_times[i] is in [window_start, t]
        in_window = (rate_times .>= window_start) .& (rate_times .<= t)

        # Compute mean of instantaneous rates in this window
        rates_in_window = inst_rates[in_window]

        if length(rates_in_window) > 0
            mean_rates[j] = mean(rates_in_window)
            variance_rates[j] = var(rates_in_window)
        else
            # No spikes in window - could use NaN or 0
            mean_rates[j] = 0.0
        end
    end

    return mean_rates, variance_rates
end

########### correlation function fits ##########

function linearmodel_XY1(list_of_dataframes, neuronid; levels=noise_levels)
    Df = []
    for nl in 1:length(levels)
        df = list_of_dataframes[nl]
        Df = vcat(Df, df[neuronid, :])
    end
    individual_neuron_data = DataFrame(Df)
    X = vcat(individual_neuron_data.avgCa...)
    Y1 = vcat(individual_neuron_data.meanR...)
    Y1 = replace(Y1, Inf => 0.0)

    data = DataFrame(X=Float64.(X), Y1=Float64.(Y1))
    model_XY1 = lm(@formula(Y1 ~ X), data)
    return model_XY1
end

function linearmodel_XY2(list_of_dataframes, neuronid; levels=noise_levels)
    Df = []
    for nl in 1:length(levels)
        df = list_of_dataframes[nl]
        Df = vcat(Df, df[neuronid, :])
    end
    individual_neuron_data = DataFrame(Df)
    X = vcat(individual_neuron_data.avgCa...)
    Y2 = vcat(individual_neuron_data.σ2ᵣ...)
    Y2 = replace(Y2, NaN => 0.0) #  check this 

    data = DataFrame(X=Float64.(X), Y2=Float64.(Y2))
    model_XY2 = lm(@formula(Y2 ~ X), data)
    return model_XY2
end


########### misc. ##########

function iterate_pairs(x_range::AbstractVector, y_range::AbstractVector; order::Symbol=:x_first)
    pairs = if order == :x_first
        [(x, y) for x in x_range, y in y_range]
    elseif order == :y_first
        [(x, y) for y in y_range, x in x_range]
    end
    return vec(pairs)  # Flatten the 2D array of tuples into a vector
end

function ensure_dir_exists(dirname::String)
    dirpath = joinpath(pwd(), dirname)
    if !isdir(dirpath)
        mkdir(dirpath)
        println("Directory created: $dirpath")
    else
        println("Directory already exists: $dirpath")
    end
end

#zeroinput = zeros(length(time_vec))