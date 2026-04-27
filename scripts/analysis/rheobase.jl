"""
Find rheobase current that makes system lose stable (quiescent) equilibrium in CS model
"""

function detect_firing(gparams, I)
    p = (I, gparams)
    ic0 = Float64[-55.0, B_cs(-55.0)...]
    prob = ODEProblem(F_inplace!, ic0, (0, 1000), p)
    sol = OrdinaryDiffEq.solve(prob, Tsit5(); dt=0.01, saveat=0.1)

    spikes = findall(x -> x > 0.0, sol[1, :])
    return length(spikes) > 0
end

function find_rheobase(I_min, I_max)
    rhb = []
    Nsweep = 20
    Is = range(I_min, I_max; length=Nsweep)
    for el in eachrow(SilentBank)
        gparams = Tuple(el)
        found = false
        i=1
        while !found && i <= length(Is)
            I = Is[i]
            found = detect_firing(gparams, I)
            i += 1
        end
        rhb_val = found ? Is[i-1] : NaN
        push!(rhb, rhb_val)     
    end
    return rhb
end

# gparams = Tuple(SilentBank[1, :])
# sol = single_CS(gparams, pinknoise_input[(4.0f0, 1.0f0)][1], tspan, time_sol)
# plot(sol; idxs=1)