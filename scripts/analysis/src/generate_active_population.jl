"""
DICs are used to generate populations of neurons with high bursting activity or 
spiking tonic activity. A fixed subset of conductances is randomly chosen and the 
rest are solved for given target DIC values. 
"""

function compensateDICs(gf_vth, gs_vth, gu_vth, vth)
    #generate random n-3 conductances and compensate for desired DICs with remaining 3
    #gf_vth, gs_vth, gu_vth are the desired DICs

    CaS_max = 35.0
    CaT_max = 10.0
    KCa_max = 600.0
    Kd_max = 150.0

    CaS_min = 6.0
    CaT_min = 1.0
    KCa_min = 30.0
    Kd_min = 50.0

    gleak = 0.02 + 0.05 * rand()
    gCaS = gleak * (CaS_min + CaS_max * rand()) / 0.01
    gCaT = gleak * (CaT_min + CaT_max * rand()) / 0.01
    gKCa = gleak * (KCa_min + KCa_max * rand()) / 0.01
    gKd = gleak * (Kd_min + Kd_max * rand()) / 0.01

    fixed_ionchannels = (CaS([gCaS]), CaT([gCaT]), KCa([gKCa]), Kdr([gKd]))

    find_ionchannels = (Na([1.0]), Ka([1.0]), H([1.0]))


    #create linear system of equations
    gf_vth_sens = [Addend(channel, find_ionchannels, 1, vth, true) for channel in find_ionchannels]
    gs_vth_sens = [Addend(channel, find_ionchannels, 2, vth, true) for channel in find_ionchannels]
    gu_vth_sens = [Addend(channel, find_ionchannels, 3, vth, true) for channel in find_ionchannels]
    A = Matrix(
        [(gf_vth_sens ./ gleak)'
        (gs_vth_sens ./ gleak)'
        (gu_vth_sens ./ gleak)']) #contribution of unknown conductances to DICs


    B = zeros(3, 1) #contribution of known conductances to DICs
    B[1, 1] = gf_vth - (1 / gleak) * sum(Addend(channel, fixed_ionchannels, 1, vth, false) for channel in fixed_ionchannels)
    B[2, 1] = gs_vth - (1 / gleak) * sum(Addend(channel, fixed_ionchannels, 2, vth, false) for channel in fixed_ionchannels)
    B[3, 1] = gu_vth - (1 / gleak) * sum(Addend(channel, fixed_ionchannels, 3, vth, false) for channel in fixed_ionchannels)

    #compensate with Na, Ka, H conductances
    gNa, gKa, gH = A \ B

    return [gNa, gCaS, gCaT, gKa, gKCa, gKd, gH, gleak]
end

zerofunc(t) = 0.0

function generate_Bursty_population(num_neurons::Int64, gf, gs, gu)
    bursty_population = []

    for i in 1:num_neurons
        g = compensateDICs(gf, gs, gu, -48.0)
        spiketimes = find_spiketimes(vec(simulate_neuron(Tuple(g), [zerofunc])[1]))
        b = try
            Burstiness(spiketimes)
        catch err
            NaN
        end
        if !(isnan(b)) && b > 0.0
            append!(bursty_population, [g])
        end
    end
    return bursty_population
end

function generate_Tonic_population(num_neurons::Int64, gf, gs, gu)
    tonic_population = []

    for i in 1:num_neurons
        g = compensateDICs(gf, gs, gu, -48.0)
        spiketimes = find_spiketimes(vec(simulate_neuron(Tuple(g), [zerofunc])[1]))
        b = try
            Burstiness(spiketimes)
        catch err
            NaN
        end
        if !(isnan(b)) && b == 0.0
            append!(tonic_population, [g])
        end
    end
    return tonic_population
end

function generate_populations()

    populationB1 = generate_Bursty_population(150, 5.0, 2.1, 0.8)
    populationB2 = generate_Bursty_population(150, 6.4, 1.7, 1.0)
    populationB3 = generate_Bursty_population(150, 1.5, 2.7, 0.4)
    populationB4 = generate_Bursty_population(150, 1.2, 2.7, 0.1)
    populationB5 = generate_Bursty_population(150, 7.5, 1.0, 0.6)
    populationB6 = generate_Bursty_population(150, 7.5, 2.5, 0.5)

    populationBursters = vcat(populationB1, populationB2, populationB3,
                        populationB4, populationB5, populationB6)

    CSV.write("STG/Burstingconductances.csv",
    DataFrame(mapreduce(permutedims, vcat, populationBursters),
             [:gNa, :gCaS, :gCaT, :gKa, :gKCa, :gKd, :gH, :gleak]))

    populationT1 = generate_Tonic_population(200, 5.0, -1.0, 0.5)
    populationT2 = generate_Tonic_population(200, 4.0, -1.0, 0.6)
    populationT3 = generate_Tonic_population(200, 3.0, -1.0, 0.7)
    populationT4 = generate_Tonic_population(200, 3.0, -1.0, 1.0)
    populationT5 = generate_Tonic_population(200, 5.0, -1.0, 1.0)
    populationT6 = generate_Tonic_population(200, 6.0, -1.5, 1.2)

    populationTonics = vcat(populationT1, populationT2, populationT3,
                        populationT4, populationT5, populationT6)

    CSV.write("STG/Tonicconductances.csv",
        DataFrame(mapreduce(permutedims, vcat, populationTonics), [:gNa, :gCaS, :gCaT, :gKa, :gKCa, :gKd, :gH, :gleak]))

end