#makes a module

module RateStats

using OrdinaryDiffEq, Random, LinearAlgebra, ForwardDiff, DataFrames
using SciMLStructures, JLD2, Statistics, Symbolics, FFTW, GLM

include("stg_odes.jl")
export neuron_ode!, neurons_ode!, neuron_syn_ode!, neurons_syn_ode!, τav, τCa, u0
export neuron_syn_ode!, neurons_syn_ode!, A_stg, B_stg, E_Na, E_K, E_L, E_H, DynECa, Cm, factorarea

include("helper_functions.jl")
export find_spiketimes, Burstiness, bursts, isi, insta_rate, counting_rate
export nanmean, nanvar, linearmodel_XY1, linearmodel_XY2, compute_running_ratestats

include("simulation_functions.jl")
export dt, dtinv, transient, tspan, time_vec, time_sol, Ti, Tf, num_trials, μᵢ, σᵢ, noise_levels, pinknoise_input
export simulate_neuron, readout, generate_pink_noise, ramp_pink_noise, single_trial, mean_levels, rampvariance_CS, rheobases, bases

include("DICs.jl")
export fastDIC, slowDIC, ultraslowDIC, Gate_properties, Addend, ∇I∞_dXinf, dXinf_dV, inputconductance
export Na, CaS, CaT, Ka, KCa, Kdr, H, leak, get_gs, make_channels

include("generate_active_population.jl")
export compensateDICs, generate_Bursty_population, generate_Tonic_population

# include("ConnorStevens_odes.jl")
# export CSneuron_ode!, CSneurons_ode!, u0_CS, V_idxs_CS, Ca_idxs_CS, Avg_idxs_CS, simulate_CSneuron, single_CS

end