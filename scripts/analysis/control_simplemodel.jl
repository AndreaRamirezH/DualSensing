"""
Integral control applied to the simple model where x=x(gsyn,muapp)
"""

function control_single_g!(du, u, p, t)
    
    input = p[1]
    dt = p[2]
    τ_g = p[3]
    γ = p[4]
    μ⁺, σ2⁺ = p[5]
    ϵ1, α = p[6]
    ϵ2, β = p[7]
    model = p[8]
    γ0, γ1, γ2, γ3 = coef(model)

    timeidx = Int(clamp(round(t / dt) + 1, 1, length(input)))
    gsyn_nominal = 1.0
    Δgsyn = u[1]
    domain_coordinates = DataFrame(X1=[gsyn_nominal + Δgsyn], X2=[input[timeidx]])
    predicted_acc = predict(model, domain_coordinates)[1]

    du[1] = (γ / τ_g) * ((σ2⁺-ϵ2) - β * predicted_acc) / σ2⁺
end

function control_simple_model(input, τ_g, γ, (μ⁺, σ2⁺), (ϵ1, α), (ϵ2, β), model)
    init_cond = [0.0]
    p = (input, dt, τ_g, γ, (μ⁺, σ2⁺), (ϵ1, α), (ϵ2, β), model)
    prob = ODEProblem(control_single_g!, init_cond, tspan_long, p)
    sol = OrdinaryDiffEq.solve(prob, Tsit5(), dt=dt; saveat=tspan_long[1]:dt:tspan_long[2])
    return sol
end

###########################################################################

f_gsyn = jldopen("data/raw/STG/control/biophysical_gsyn_rampvariance.jld2","r")
sol_full = f_gsyn["sol_full"]
Δgsyn_full = sol_full[15, :]
p_gsyn = f_gsyn["p"]
τ_g, γ = p_gsyn[4], p_gsyn[7]

neuronid = 136
conductances = Vector(ActiveBank[neuronid, :])

model_XY1 = linearmodel_XY1(list_stgvariance_dataframes, neuronid)
model_XY2 = linearmodel_XY2(list_stgvariance_dataframes, neuronid)
ϵ1, α = coef(model_XY1)
ϵ2, β = coef(model_XY2)
ĥ₋(r̄, α, β, ϵ1, ϵ2) = (β * r̄) / α + ϵ2 - (β * ϵ1) / α #function that goes from mean to variance

#need to create a pink noise input where input variance is ramped
number_changes = 12
tspan_long = (tspan[1], number_changes * τ_g)
time_vec_long = tspan_long[1]:dt:tspan_long[2]
n = length(tspan[1]:dt:number_changes * τ_g)
input_conditions = range(noise_levels[2],noise_levels[1+number_changes],length=n)

μ⁺ = 23.0f0
σ2⁺ = ĥ₋(μ⁺, α, β, ϵ1, ϵ2) #variance target based on mean target
target_acc = (μ⁺ - ϵ1) / α #target for average calcium
sol = control_simple_model(input_conditions, τ_g, γ, (μ⁺, σ2⁺), (ϵ1, α), (ϵ2, β), model)
Δgsyn_simple = sol[1,:]

domain_coordinates = DataFrame(X1=1.0 .+ Δgsyn_simple, X2=input_conditions)
predicted_acc = predict(model, domain_coordinates)

jldsave("data/raw/STG/control/simple_gsyn_rampvariance.jld2"; 
sol_simple=sol, predicted_acc = predicted_acc, targets = (μ⁺, σ2⁺, target_acc),  input_conditions = input_conditions)
