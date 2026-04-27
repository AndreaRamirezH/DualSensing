
"""
Connor JA, Walter D, McKown R (1977)
"""
# Voltage shifts
MSHFT = -5.3
HSHFT = -12.0
NSHFT = -4.3

# Alpha and beta for mNa
alpha_m(Vm) = (-0.1 * (Vm + 35 + MSHFT))/(exp(-(Vm + 35 + MSHFT) / 10) - 1)
beta_m(Vm) = 4 * exp(-(Vm + 60 + MSHFT) / 18)
m∞Na(Vm) = alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))
τmNa(Vm) = (1 / 3.8) * (1 / (alpha_m(Vm) + beta_m(Vm)))

# Alpha and beta for hNa
alpha_h(Vm) = 0.07 * exp(-(Vm + 60 + HSHFT) / 20)
beta_h(Vm) = 1 / (exp(-(Vm + 30 + HSHFT) / 10) + 1)
h∞Na(Vm) = alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))
τhNa(Vm) = (1 / 3.8) * (1 / (alpha_h(Vm) + beta_h(Vm)))

# mCa
m∞Ca(V) = 1.0f0 / (1.0f0 + exp((V + 50) / -0.15))
τmCa(V) = 2.35 #ms

# Alpha and beta for mKd
alpha_n(Vm) = -0.01 * (Vm + 50 + NSHFT) / (exp(-(Vm + 50 + NSHFT) / 10) - 1)
beta_n(Vm) = 0.125 * exp(-(Vm + 60 + NSHFT) / 80)
m∞Kdr(Vm) = alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))
τmKdr(Vm) = (2 / 3.8) * (1 / (alpha_n(Vm) + beta_n(Vm)))

# Alpha and beta for mKa, hKa
m∞Ka(Vm) = (0.0761 * exp((Vm + 94.22) / 31.84) / (1 + exp((Vm + 1.17) / 28.93)))^(1 / 3)
τmKa(Vm) = 0.3632 + 1.158 / (1 + exp((Vm + 55.96) / 20.12))
h∞Ka(Vm) = 1 / (1 + exp((Vm + 53.3) / 14.54))^4
τhKa(Vm) = 1.24 + 2.678 / (1 + exp((Vm + 50) / 16.027))

const Cm = 10.0f0 # specific capacitance cₘ is a biological constant ~ 10 nF/mm^2
const τCa = 20.0f0 #20 ms
const factorarea = 14.96f0 * 0.0628f0 #from Liu et al '98
const τav = 6000.0f0

const Eq_Na = 55.0f0
const Eq_K = -75.0f0
const Eq_L = -17.0f0
const Eq_Ca = 120.0f0

A_cs(v) = diagm(1 ./ [τmNa(v), τhNa(v), τmCa(v), τmKa(v), τhKa(v), τmKdr(v)])
B_cs(v) = [m∞Na(v), h∞Na(v), m∞Ca(v), m∞Ka(v), h∞Ka(v), m∞Kdr(v)]

function CSneuron_ode!(du,u,p,t)
    gNa, gCa, gKa, gKd, gleak = p[1]
    inp = p[2]
    dt = p[3]
    timeidx = Int(clamp(round(t / dt) + 1, 1, length(inp)))

    v,ca,avg = u[1:3]
    x = u[4:end]
    mNa, hNa, mCa, mKa, hKa, mKd = x

    du[1] = sum([gNa * mNa^3 * hNa^1 * (Eq_Na - v),
            gCa * mCa^2 * (Eq_Ca - v),
            gKa * mKa^3 * hKa^1 * (Eq_K - v),
            gKd * mKd^4 * (Eq_K - v),
            gleak *(Eq_L - v),
            inp[timeidx]
            ]) /Cm

    du[2] = (1 /τCa) * (-ca + 0.05 + (factorarea * (gCa * mCa^2 * (Eq_Ca - v)) / Cm))
    du[3] = (1 / τav) * (-avg + ca)

    du[4:end] .= A_cs(v) * (B_cs(v) - x)
end

function CSneurons_ode!(du, u, p, t)
    conductances = p[1]
    inputs = p[2]
    dt = p[3]
    num_trials = length(inputs)

    #call neuron_ode
    for i = 1:num_trials
        dz = @view du[1+(i-1)*9:9*i]
        z = @view u[1+(i-1)*9:9*i]
        CSneuron_ode!(dz, z, (conductances, inputs[i], dt), t)
    end

end

u0_CS = [-55.0f0, 0.05f0, 1.0f0, B_cs(-55.0f0)...]; #initial conditions for a single neuron

const V_idxs_CS = [1 + (i - 1) * 9 for i = 1:num_trials] #indices of V in solution 
const Ca_idxs_CS = [2 + (i - 1) * 9 for i = 1:num_trials] #indices of Ca in solution
const Avg_idxs_CS = [3 + (i - 1) * 9 for i = 1:num_trials] #indices of Avg in solution


function single_CS(conductances, input, tspan, time_sol)
    p = (conductances, input, dt)
    prob_ = ODEProblem(CSneuron_ode!, u0_CS, tspan, p)
    sol_ = OrdinaryDiffEq.solve(prob_, Tsit5(), dt=dt; save_idxs=[1, 2, 3], saveat=time_sol)
    v_traces = Float32.(sol_[1, :])
    return sol_
end

function simulate_CSneuron(conductances::Tuple, inputs::Vector; tspan=tspan, dt=dt::Float32, u0=u0_CS::Vector)
    num_trials = length(inputs)
    init_cond = repeat(u0, num_trials) #initial conditions for each trial
    p = (conductances, inputs, dt)
    prob_trials = ODEProblem(CSneurons_ode!, init_cond, tspan, p)

    savevars = vcat(V_idxs_CS, Ca_idxs_CS, Avg_idxs_CS) #indices of V, Ca, Avg in solution

    sol_trials = OrdinaryDiffEq.solve(prob_trials, Tsit5(), dt=dt; save_idxs=savevars, saveat=time_sol)
    v_traces = Float32.(sol_trials[1:num_trials, :])
    ca_traces = Float32.(sol_trials[1+num_trials:2*num_trials, :])
    avg_traces = Float32.(sol_trials[1+2*num_trials:3*num_trials, :])
    return v_traces, ca_traces, avg_traces
end

# gNa, gCa, gKa, gKd, gleak = [120, 1, 210, 20, 0.3] in https://www.pnas.org/doi/epdf/10.1073/pnas.1516400112