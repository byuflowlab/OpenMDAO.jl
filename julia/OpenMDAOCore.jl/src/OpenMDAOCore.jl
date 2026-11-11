module OpenMDAOCore

using ADTypes: ADTypes
using ComponentArrays: ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata, indexmap
using DifferentiationInterface: DifferentiationInterface
using Random: rand!

include("utils.jl")
export get_rows_cols, ca2strdict, rcdict2strdict, get_rows_cols_dict, ca2strdict_sparse
export can_jvp, can_vjp

# `PerturbedDenseSparsityDetector` (type + constructor) is declared in the main
# package so it can be imported without the extension loaded. The sparse
# *functionality* (the `SparseFlavor` constructors, `compute_partials!`,
# sparsity detection, `ca2strdict_sparse`, `get_rows_cols_dict_from_sparsity`,
# the sparse overloads of `_maybe_nonzeros`, etc.) is provided by the
# `OpenMDAOCoreSparseMatrixColoringsExt` extension, which loads when both
# `SparseArrays` and `SparseMatrixColorings` are available.
export PerturbedDenseSparsityDetector


include("interface.jl")
export AbstractComp, AbstractExplicitComp, AbstractImplicitComp
export OpenMDAOCoreError, OpenMDAOMethodError, OpenMDAOPrepNotCreatedError
export has_setup_partials
export has_compute_partials, has_compute_jacvec_product
export has_apply_nonlinear, has_solve_nonlinear, has_linearize, has_apply_linear, has_solve_linear, has_guess_nonlinear 

include("var_data.jl")
export VarData

include("partials_data.jl")
export PartialsData

include("abstract_ad.jl")
export get_callback, get_func, get_params, get_input_ca, get_output_ca, get_jacobian_ca, get_units, get_backend, get_backend_fwd, get_backend_rev, get_prep
export ADExplicitComp, ADImplicitComp,
    DerivativeFlavor, AssembledFlavor, MatrixFreeFlavor,
    DenseFlavor, SparseFlavor,
    MatrixFreeForwardFlavor, MatrixFreeReverseFlavor, MatrixFreeForwardReverseFlavor,
    DenseDerivPrep, MatrixFreeDerivPrep, MatrixFreeForwardReverseDerivPrep, SparseDerivPrep

include("dense_ad.jl")

include("matrix_free_ad.jl")
export get_dinput_ca, get_doutput_ca

include("dense_ad_implicit.jl")
include("matrix_free_ad_implicit.jl")

include("create_component.jl")
export create_explicit_component, create_implicit_component

end # module