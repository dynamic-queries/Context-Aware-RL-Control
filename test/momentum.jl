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
    h4 = plot(z,W,xlabel="Z",label="W",title="Recovery energy")
    plot(h1,h2,h3,h4,size=(1000,1000))
end

function fm1()
    
end