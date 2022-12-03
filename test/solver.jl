using OrdinaryDiffEq
using Plots
using LinearAlgebra
using CSV
using DataFrames
using Interpolations

# Utils
function Base.float(str::String15)
    if str != "0"
        man,expo = split(str,"E")
        return parse(Float64,string(man,"e",expo))
    else
        return 0.0
    end 
end 


# https://research-groups.usask.ca/tpp/documents/on_the_mechanism_of_ionization_oscillations_in_hall_thrusters__reduced_model_paper___jap_revised_.pdf
## Degrees of Freedom in the system
#1 Neutral density : Na
#2 Plasma and electron density : Np
#3 Plasma velocity : Vi
#4 Energy : Te
#5 Electron velocity : Ve

## Constants
e = 1.6e-19 # C 
mi = 131.293*1.66e-27 # kg
me = 9.1e-31 # kg
Vn = 150 # m/s
kb = 1.38e-23 # J/K
U0 = 20 # V


# Equations of evolution
function D(u,ui,dx)
    return (ui-u)/dx
end 

# Neutral contibuity equation
function f1(Na,Np,β,dx,Te)
    n = length(Na)
    du = zero(Na)
    for i=2:n
        du[i] = -Vn*D(Na[i],Na[i-1],dx) - β(Te[i])*Na[i]*Np[i]
    end 
    return du
end 

# Plasma continuity equation
function f2(Na,Np,Vi,β,dx,Te)
    n = length(Na)
    du = zero(Na)
    for i=2:n
        du[i] = -D(Np[i]*Vi[i],Np[i-1]*Vi[i-1],dx) + β(Te[i])*Na[i]*Np[i]
    end 
    return du
end 

# Plasma momentum equation
function f3(Na,Np,Vi,β,Ve,Te,dx)
    n = length(Na)
    du = zero(Na)
    for i=2:n
        du[i] = -Vi[i]*D(Vi[i],Vi[i-1],dx) + β(Te[i])*Na[i]*(Vn-Vi[i]) - (e/(mi*Np[i]))*D(Np[i]*Te[i],Np[i-1]*Te[i-1],dx) - ((e*Ve[i])/(mi*μ[i]))
    end 
    return du
end 

# Electron energy equation
function f4(Na,Np,Ve,Te,μ,K,W,dx)
    n = length(Na)
    du = zero(Na)
    for i=2:n-1
        diff = 2.5* D(μ[i+1]*Np[i+1]*Te[i+1]*D(Te[i+1],Te[i],dx),μ[i]*Np[i]*Te[i]*D(Te[i],Te[i-1],dx),dx)
        du[i] = -(5/3)*D(Np[i]*Ve[i]*Te[i],Np[i-1]*Ve[i-1]*Te[i-1],dx) 
                - Np[i]*Na[i]*K(Te[i]) - Np[i]*W[i]
                - Ve[i]*D(Np[i]*Te[i],Np[i-1]*Te[i-1],dx) 
                - ((Np[i]*Ve[i]^2)/μ[i]) 
                + diff
    end 
    return du
end

# Constitutive relation
function f5(Np,Ve,Te,Vi,μ,dx)
    n = length(Ve)
    du = zero(Ve)
    J = (U0 .+ sum((Vi[1:end-1] ./ μ[1:end-1]) .+ (D.(Np[2:end] .* kb .* Te[2:end],Np[1:end-1] .* kb .* Te[1:end-1],dx)),dims=1)) / sum(inv.(e .* Np .* μ))
    J = J[1]
    for i=2:n
        du[i] = -Ve[i] + Vi[i] - (J/(e*Np[i]))  
    end 
    return du
end 

# Consolidated forcing function
function hall_thruster!(du,u,p,t)
    Na,Np,Vi,Te,Ve = u[:,1],u[:,2],u[:,3],u[:,4],u[:,5]
    dx,β,μ,K,W = p
    du[:,1] = f1(Na,Np,β,dx,Te)
    du[:,2] = f2(Na,Np,Vi,β,dx,Te)
    du[:,3] = f3(Na,Np,Vi,β,Ve,Te,dx)
    du[:,4] = f4(Na,Np,Ve,Te,μ,K,W,dx)
    du[:,5] = f5(Np,Ve,Te,Vi,μ,dx)
    nothing
end 


# Initial and boundary conditions 
function icbc!(state::Matrix)

end 


# Simulation parameters
Nz = 50 # no unit
zmin = 0.0e-2 # m 
zmax = 5.0e-2 # m
z = zmin:(zmax-zmin)/Nz:zmax # m


## Maintain DOFs as a vector of vectors
state = rand(length(z),5)
icbc!(state)

# Mass matrix
Id = I(5)
M = I(5)
M[5,5] = 0.0
Mb = kron(I(length(z)),M)

# Parameters
function ionization_coeff()
    filename =  "test/ionization_constants.csv"
    file = CSV.read(filename,DataFrame)
    Te = float.(file.Te)
    ki_vals = file.ki
    K_vals = float.(file.K)
    ki = cubic_spline_interpolation(0.0:150.0, ki_vals)
    K = cubic_spline_interpolation(0.0:150.0,K_vals)
    return ki,K
end

function mobility(B,Np)
    ω = e*B/me
    ν = 2.5e13 * Np +  1e7.*vcat(0.2 .* ones(floor(Int,length(z)/2)),zeros(floor(Int,length(z)/2)+1)) + vcat(0.1 .* ones(floor(Int,length(z)/2)),ones(floor(Int,length(z)/2)+1)) *e .*B / (16*me)
    μ = vec((e/(me .* ν)) .* (1/(1 .+ (ω./ν).^2)))
    return μ
end 

function anomalous(Te)
    Te = state[4]
    νe = vcat(1e7*ones(floor(Int,length(z)/2)),0.4e7ones(floor(Int,length(z)/2)+1))
    ε = Te*1.5
    W = νe .* ε .* exp.(-U0./ε)
    return W
end 

## Magnetic field
Bmax = 0.015 # T
l = zmax/2 # m
δ = vcat(1.1e-2 * ones(floor(Int,length(z)/2)),1.8e-2*ones(floor(Int,length(z)/2)+1)) # m
B = Bmax*exp.(-0.5*((z.-l)./(δ)).^2) # T

dz = z[2]-z[1]
β,K = ionization_coeff()
μ = mobility(B,state[:,2])
W = anomalous(state[:,4])
p = [dz,β,μ,K,W]

du = zero(state)
hall_thruster!(du,state,p,0.0)
du

# ODE problem
tspan = (0.0,1e-3)
tsave = tspan[1]:1e-4:tspan[2]
forcing_function = ODEFunction(hall_thruster!,mass_matrix=Mb)
prob = ODEProblem(forcing_function,state,tspan,p)
sol = solve(prob,Rodas3())