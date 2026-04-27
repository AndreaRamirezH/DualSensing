using Distributed
addprocs(30)

@everywhere begin

    using Pkg
    Pkg.activate(".")
    include("setup.jl")

    function make_inputdriven_population(run_no, medium_inputs)

        plt = true # disable if no plots wanted
        all_rows = []
        zerofunc = zeros(length(time_vec))
        x_formatted(x) = x / 10

        Ca_min, Ca_max = 0.0, 2.0
        Ka_min, Ka_max = 0.0, 210.0
        Na_min, Na_max = 140.0, 600.0

        gleak = 0.3
        gNa = (Na_min + (Na_max - Na_min) * rand())
        gCa = (Ca_min + (Ca_max - Ca_min) * rand())
        gKa = (Ka_min + (Ka_max - Ka_min) * rand())
        gKd = 20.0
        conductances = Tuple(Float32.([gNa, gCa, gKa, gKd, gleak]))

        v_trace_noinp = single_CS(conductances, zerofunc; tspan=(0.0f0, transient + 2 * τav))
        spk_times_noinp = find_spiketimes(v_trace_noinp)

        v_trace_medinp = single_CS(conductances, medium_inputs[1]; tspan=(0.0f0, transient + 2 * τav))
        spk_times_medinp = find_spiketimes(v_trace_medinp)

        if length(spk_times_noinp) == 0 && prod(v_trace_noinp .< -5.0) && length(spk_times_medinp) > 1
            push!(all_rows,
                NamedTuple{(:gNa, :gCa, :gKa, :gKd, :gleak)}(Tuple(conductances)))
            df = DataFrame(all_rows)
            save("temp/silentneur_$(run_no).jld2", "df", df)
        end
        if plt 
            p = plot(v_trace, xlims=(100000, 110000), ylims=(-65, 30), xlabel="t (ms)",
                xformatter=x -> x_formatted(x), label="", ylabel="mV", legend=:outerright)
            savefig(p, "temp/$(run_no)_spiked.pdf")
        end
    
    end

    const medium_inputs = pinknoise_input[(5.0f0, 1.0f0)]


end #ends everywhere
pmap(i -> make_inputdriven_population(i, medium_inputs), 1:2000)


files = filter(x->occursin("jld2",x),readdir("ConnorStevens/temp/"))
collecteddfs = [load("ConnorStevens/temp/$f", "df") for f in files]
CSBank = reduce(vcat, collecteddfs)
CSV.write("ConnorStevens/CSconductances.csv", CSBank)