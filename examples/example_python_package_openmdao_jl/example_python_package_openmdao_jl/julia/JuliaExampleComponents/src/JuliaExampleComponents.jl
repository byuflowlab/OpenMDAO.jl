module JuliaExampleComponents

using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using ForwardDiff: ForwardDiff
using SparseMatrixColorings: SparseMatrixColorings
using NLsolve: nlsolve
using LineSearches: BackTracking
using ImplicitAD: implicit

include("paraboloid.jl")
export get_parabaloid_comp

include("circle.jl")
export get_arctan_yox_comp, get_circle_comp, get_r_con_comp, get_theta_con_comp, get_delta_theta_con_comp, get_l_conx_comp

include("circuit.jl")
export MyCircuit, circuit_solve

end # module JuliaParaboloidComponent
