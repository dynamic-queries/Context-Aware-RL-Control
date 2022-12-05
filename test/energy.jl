include("solver.jl")
using LSODA
using ForwardDiff

begin
    # Simulation parameters
    Nz = 100 # no unit
    zmin = 0.0e-2 # m 
    zmax = 5.0e-2 # m
    z = zmin:(zmax-zmin)/Nz:zmax # m


    ## Maintain DOFs as a vector of vectors
    state = zeros(length(z),5)
    icbc!(state,z)

    g1 = plot(z,state[:,1],xlabel="Z",label="Na",title="Neutral density")
    g3 = plot(z,state[:,2],xlabel="Z",label="Np",title="Plasma density")
    g4 = plot(z,state[:,3],xlabel="Z",label="Vi",title="Ion velocity")
    g2 = plot(z,state[:,4],xlabel="Z",label="Te",title="Electron temperature")
    g5 = plot(z,state[:,5],xlabel="Z",label="Ve",title="Electron velocity")
    plot(g1,g2,g3,g4,g5,size=(1000,1000))
end

begin
    ## Magnetic field
    Bmax = 0.015 # T
    l = zmax/2 # m
    δ = vcat(1.1e-2 * ones(floor(Int,length(z)/2)),1.8e-2*ones(floor(Int,length(z)/2)+1)) # m
    B = Bmax*exp.(-0.5*((z.-l)./(δ)).^2) # T

    dz = z[2]-z[1]
    β,K = ionization_coeff()
    μ = mobility(B,state[:,1])
    W = anomalous(state[:,4])
    p = [dz,β,μ,K,W]

    h1 = plot(z,β.(state[:,4]),xlabel="Z",label="β",title="Ionization coefficient")
    h2 = plot(z,K.(state[:,4]),xlabel="Z",label="K",title="Ionization energy")
    h3 = plot(z,μ,xlabel="Z",label="μ",title="Electron mobility")
    h4 = plot(z,W,xlabel="Z",label="W",title="Anomalous energy")
    plot(h1,h2,h3,h4,size=(1000,1000))
end

function fe1(Np,Na,K,W,Te,ϕ,Ve,dx)
    n = length(Np)
    du = zero(Np)
    for i=2:n-1
        # -Np[i]*Na[i]*K(Te[i])
        a1 =  -Np[i]*W[i]
        a2 = -Np[i]*Ve[i]*D(ϕ[i],ϕ[i-1],dx)
        a3i = -(5/3)*Np[i]*Ve[i]*kb*Te[i]
        a3im = -(5/3)*Np[i-1]*Ve[i-1]*kb*Te[i-1]
        a3 = D(a3i,a3im,dx)
        a4i = (10/9)*μ[i+1]*Np[i+1]*kb*Te[i+1]*D(kb*Te[i+1],kb*Te[i],dx)
        a4im = (10/9)*μ[i]*Np[i]*kb*Te[i]*D(kb*Te[i],kb*Te[i-1],dx)
        a4 = D(a4i,a4im,dx)
        du[i] = a1 + a2 + a3 +a4
    end 
    return du
end 

ϕ = 2 .- ((2)/(zmax)) .* z
function forcing_function(du,u,p,t)
    dx,β,μ,K,W = p
    Na,Np,Vi,Te,Ve = state[:,1],state[:,2],state[:,3],u ./state[:,2],state[:,5]
    du .= fe1(Np,Na,K,W,Te,ϕ,Ve,dx)
end 

tspan = (0.0,1)
u0 = Np .* Te
# Boundary conditions
u0[1] = 3 .*Np[1]
u0[end] = 3 .*Np[1]
prob = ODEProblem(forcing_function,u0,tspan,p)
sol = solve(prob,Tsit5())
solution = Array(sol) 

anim = @animate for i=1:7
    plot(solution[1:end-10,i] ./ Np[1:end-10],title="Timestep : $(i)")
end 
gif(anim,"neutral_density.gif",fps=3)