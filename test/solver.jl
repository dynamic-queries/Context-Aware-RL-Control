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
    return (u-ui)/dx
end 

function D2(ui,uim1,uip1,dx)
    return (uip1+uim1-2*ui)/dx^2
end 

# Neutral contibuity equation
function f1(Na,Np,β,dx,Te)
    n = length(Na)
    du = zero(Na)
    for i=2:n
        du[i] = -Vn * D(Na[i],Na[i-1],dx) - β(Te[i])*Na[i]*Np[i]
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

# Electron energy equation  - faulty
function f4(Na,Np,Ve,u,μ,K,W,dx)
    n = length(Na)
    du = zero(Na)
    t = u ./ Np
    for i=2:n-1
        fa = -(5/3)*D(u[i]*Ve[i],u[i-1]*Ve[i-1],dx)
        fb = +(5/3)*(kb/e)*(D(μ[i]*u[i],μ[i-1]*u[i-1],dx) .* D(t[i],t[i-1],dx) .+ (μ[i]*u[i]) .* D2(t[i],t[i-1],t[i+1],dx))
        fc = +(2/3)*Ve[i]*D(u[i],u[i-1],dx)
        # fd = -(2/(3*kb))*(Np[i]*Na[i]*K(t[i]))
        # fe = -(2/(3))*(Np[i]*W[i]/kb)
        # ff = (2/3)*(e/kb)*(Np[i]*Ve[i]^2)/μ[i]
        du[i] = fa + fb + fc #+ fd + fe + ff
    end 
    display(plot(du))
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

# Consolidated forcing function formulating a DAE problem
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
function icbc!(state::Matrix,z::Union{Vector,StepRangeLen})
    xmean = 0.0
    σ = 0.75e-2
    state[:,1] = 1 .+ 12e18 .* exp.(-((z .- xmean)./σ).^2)
    xmean = 0.5e-2
    σ = 0.3e-2
    state[:,2] = 1 .+ 6e17 .* exp.(-((z .- xmean)./(σ)).^2)
    state[:,3] = 1 .+ 2e1*inv.((1/(exp.(-1e2*z))))
    xmean = 2e-2
    σ = 0.5e-2
    state[:,4] = 1 .+ 30*exp.(-((z.-xmean)./σ).^2)
    # state[:,5] = 1 .+ 1e4 .* (1 ./ (1 .+ exp.(-1.0e2*z))) .- 3.5e4
    state[:,5] = 1e5*exp.(-1e2 .*z)
    nothing
end 

# Parameters
function ionization_coeff()
    filename =  "test/ionization_constants.csv"
    file = CSV.read(filename,DataFrame)
    Te = float.(file.Te)
    ki_vals = file.ki
    K_vals = float.(file.K)
    ki = cubic_spline_interpolation(minimum(Te):maximum(Te), ki_vals)
    K = cubic_spline_interpolation(minimum(Te):maximum(Te),K_vals)
    return ki,K
end

function mobility(B,Na)
    ω = e*B/me
    ν = 2.5e13 * Na +  1e7.*vcat(0.2 .* ones(floor(Int,length(z)/2)),zeros(floor(Int,length(z)/2)+1)) + vcat(0.1 .* ones(floor(Int,length(z)/2)),ones(floor(Int,length(z)/2)+1)) *e .*B / (16*me)
    μ = (e./(me * ν)) .* (1 ./(1 .+ (ω./ν).^2))
    return μ
end 

function anomalous(Te)
    νe = vcat(1e7*ones(floor(Int,length(z)/2)),0.4e7ones(floor(Int,length(z)/2)+1))
    ε = Te*1.5
    W = νe .* ε .* exp.(-U0./ε)
    return W
end 
