module OpenMDAOCore

using ComponentArrays: ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata
using ForwardDiff: ForwardDiff
using SparseArrays: sparse, findnz
using SparseDiffTools: forwarddiff_color_jacobian! 

include("utils.jl")
export get_rows_cols, get_rows_cols_dict_from_sparsity, ca2strdict

include("interface.jl")
export AbstractComp, AbstractExplicitComp, AbstractImplicitComp
export has_setup_partials
export has_compute_partials, has_compute_jacvec_product
export has_apply_nonlinear, has_solve_nonlinear, has_linearize, has_apply_linear, has_solve_linear, has_guess_nonlinear 

include("var_data.jl")
export VarData

include("partials_data.jl")
export PartialsData

include("sparse_ad.jl")
export AbstractAutoSparseForwardDiffExplicitComp
export get_callback, get_input_ca, get_output_ca, get_sparse_jacobian_ca, get_sparse_jacobian_cache, get_units
export generate_perturbed_jacobian!

end # module
