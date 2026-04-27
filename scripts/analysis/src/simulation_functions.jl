
const cutoff_f = 200.0f0 # Cutoff frequency for neuron input (Hz)
const sample_rate = 2 * cutoff_f  
# Sample rate (Hz) at least 2x the highest freq. we want to represent
# dt should be less than 1/(2*cutoff)= 0.0025 seconds according to Nyquist theorem. Our dt = 0.01 ms. 

# uses FFTW to create a white noise signal in the frequency domain
function generate_white_noise(n::Int; μ::Float32=0.0f0, σ2::Float32=1.0f0)

    # Step 1: Generate raw white noise
    white = randn(Float32,n)

    # Step 2: Get frequency bins
    f = collect(rfftfreq(n, 1 / sample_rate)) #real valued FFT frequency bins

    # Step 3: Apply low-pass filter in frequency domain
    white_fft = rfft(white)
    white_fft[f.>cutoff_f] .= 0.0f0  # zero out high frequencies

    filtered = irfft(white_fft, n)

    # Step 4: Normalize to specified mean and variance
    filtered .-= mean(filtered)
    filtered ./= std(filtered)

    return μ .+ sqrt(σ2) .* filtered
end

function generate_pink_noise(n::Int; μ::Float32=0.0f0, σ2::Float32=1.0f0)
    white = randn(Float32, n) #n is number of time samples
    f = collect(rfftfreq(n, 1 / sample_rate)) #real valued FFT frequency bins
    f[1] = f[2]  # avoid divide-by-zero at DC (0Hz)

    scaling = 1.0f0 ./ sqrt.(f)
    scaling[f.>cutoff_f] .= 0.0f0  # zero out frequencies above cutoff

    white_fft = rfft(white)
    pink_fft = white_fft .* scaling
    pink = irfft(pink_fft, n) 
    #Converts the scaled frequency-domain signal back to the time domain using the inverse real FFT
    
    # Normalize to zero mean, unit variance
    pink .-= mean(pink)
    pink ./= std(pink)

    return μ .+ sqrt(σ2) .* pink
end

function ramp_pink_noise(n::Int, μ, σ2)
    white = randn(Float32, n) #n is number of time samples
    f = collect(rfftfreq(n, 1 / sample_rate)) #real valued FFT frequency bins
    f[1] = f[2]  # avoid divide-by-zero at DC (0Hz)

    scaling = 1.0f0 ./ sqrt.(f)
    scaling[f.>cutoff_f] .= 0.0f0  # zero out frequencies above cutoff

    white_fft = rfft(white)
    pink_fft = white_fft .* scaling
    pink = irfft(pink_fft, n) 
    #Converts the scaled frequency-domain signal back to the time domain using the inverse real FFT
    
    # Normalize to zero mean, unit variance
    pink .-= mean(pink)
    pink ./= std(pink)

    return μ .+ sqrt.(σ2) .* pink
end

########## Simulation time setup ##########
const num_trials = 15
const μᵢ = 0.0f0
const σᵢ = 2.0f0
const max_σᵢ = 14.0f0      # input current has units (nA/mm²)
const max_μᵢ = 10.0f0

const dt = 0.1f0 # ms (tspan is in ms)
const dtinv = 1.0f0 / dt
const transient = 1000.0f0 # ms, time to ignore at the beginning 

const tspan = (0.0f0, transient + 3 * τav)
const Ti = Int(transient * dtinv)
const Tf = Int(tspan[end] * dtinv)
const time_vec = 0.0f0:dt:tspan[2] #time vec used for generating noise inputs
const time_sol = transient:dt:tspan[2]

const mean_levels=[0.0f0, 1.0f0:max_μᵢ...]
const noise_levels = [0.1f0, 1.0f0:max_σᵢ...]

rheobases = jldopen("ConnorStevens/rheobases.jld2", "r")["rheobases"]
bases = Float32.(unique(sort(floor.(rheobases))))

pinknoise_input = Dict{Tuple{Float32,Float32},Any}()

#for m in bases
    for σᵢ in noise_levels
    value = Vector{Vector}(undef, num_trials)

    for trial in 1:num_trials
        noise_series = generate_pink_noise(length(time_vec); μ=μᵢ, σ2=σᵢ^2)
        value[trial] = noise_series
        # each element inside vector value of the dictionary is input for a different trial 
    end

    pinknoise_input[(μᵢ, σᵢ)] = value
    end
#end


function single_trial(conductances, input, tspan, time_sol; saveidxs=[1, 2, 3])
    p = (conductances, input, dt)
    prob_ = ODEProblem(neuron_ode!, u0, tspan, p)
    sol_ = OrdinaryDiffEq.solve(prob_, Tsit5(), dt=dt; save_idxs=saveidxs, saveat=time_sol)
    v_traces = Float32.(sol_[1, :])
    #return v_traces, Float32(sol_[3, end])
    return sol_
end

"""
function simulate_neuron(conductances::Tuple, inputs::Vector; tspan=tspan, dt=dt)

    Simulates a neuron for however many inputs are given. 
    Inputs is a vector of noise timeseries for num_trials trials.  
        Returns matrices of voltage and calcium traces.
"""

const V_idxs = [1 + (i - 1) * 14 for i = 1:num_trials] #indices of V in solution 
const Ca_idxs = [2 + (i - 1) * 14 for i = 1:num_trials] #indices of Ca in solution
const Avg_idxs = [3 + (i - 1) * 14 for i = 1:num_trials] #indices of Avg in solution

function simulate_neuron(conductances::Tuple, inputs::Vector; tspan=tspan, dt=dt::Float32, u0=u0::Vector{Float32})
    num_trials = length(inputs)
    init_cond = repeat(u0, num_trials) #initial conditions for each trial
    p = (conductances, inputs, dt) 
    prob_trials = ODEProblem(neurons_ode!, init_cond, tspan, p)

    savevars = vcat(V_idxs, Ca_idxs, Avg_idxs) #indices of V, Ca, Avg in solution
    
    sol_trials = OrdinaryDiffEq.solve(prob_trials, Tsit5(), dt=dt; save_idxs=savevars, saveat=time_sol)
    v_traces = Float32.(sol_trials[1:num_trials, :])
    ca_traces = Float32.(sol_trials[1+num_trials:2*num_trials, :])
    avg_traces = Float32.(sol_trials[1+2*num_trials:3*num_trials, :])
    return v_traces, ca_traces, avg_traces
end

########## output ##########

function readout_STG(NeuronBank::DataFrame, neuronid::Int, inputs::Vector, level::Int64, foldername::String)
    conductances = Tuple(Float32.(collect(NeuronBank[neuronid,:])))

    v_traces, ca_traces, avg_traces = simulate_neuron(conductances, inputs)

    rowtrial_avgCa = avg_traces[:, end] #average Ca at the end of each trial

    rowtrial_meanV = nanmean(v_traces, 2) #skips NaNs, computes mean along dim=2 
    rowtrial_varV = nanvar(v_traces, 2) 
    
    rowtrial_spktms = collect(hcat([find_spiketimes(v_traces[i, :]) for i = 1:size(v_traces)[1]]))

    rowtrial_pattern = Burstiness.(rowtrial_spktms)

    dir = "./$foldername/level$level"
    mkpath(dir)
    save("$dir/$(neuronid)neur_trials.jld2", "rowtrial_avgCa", rowtrial_avgCa, 
                                            "rowtrial_meanV", rowtrial_meanV,
                                            "rowtrial_varV", rowtrial_varV,
                                            "rowtrial_spktms", rowtrial_spktms,
                                            "rowtrial_pattern", rowtrial_pattern)
end


function readout_CS(NeuronBank::DataFrame, neuronid::Int, Inputs::Dict, fixed_μ::Float32, noise_level::Int64, foldername::String)
    σᵢ = noise_levels[noise_level]
    conductances = Tuple(Float32.(collect(NeuronBank[neuronid, :])))
    inputs = Inputs[(fixed_μ, σᵢ)]

    v_traces, ca_traces, avg_traces = simulate_CSneuron(conductances, inputs)

    tmp_row = Vector{Vector{Float32}}(undef, 5)
    for trial = 1:5
        tmp_row[trial] = v_traces[trial, 120000:130000]
    end

    rowtrial_avgCa = avg_traces[:, end] #average Ca at the end of each trial

    rowtrial_meanV = nanmean(v_traces, 2) #skips NaNs, computes mean along dim=2 
    rowtrial_varV = nanvar(v_traces, 2)

    rowtrial_spktms = collect(hcat([find_spiketimes(v_traces[i, :]) for i = 1:size(v_traces)[1]]))

    rowtrial_pattern = Burstiness.(rowtrial_spktms)

    dir = "./$foldername/noiselevel$noise_level"
    mkpath(dir)
    save("$dir/$(neuronid)neur_trials.jld2", "rowtrial_avgCa", rowtrial_avgCa,
        "rowtrial_meanV", rowtrial_meanV,
        "rowtrial_varV", rowtrial_varV,
        "rowtrial_spktms", rowtrial_spktms,
        "rowtrial_pattern", rowtrial_pattern)

    return tmp_row
end