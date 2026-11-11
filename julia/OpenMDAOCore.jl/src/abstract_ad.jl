# Derivative "flavors" — the strategy used to compute derivatives of an
# `ADExplicitComp`. Following the OpenMDAO4Core.jl design: the flavor is a
# singleton type carried as the first type parameter of `ADExplicitComp`, and
# all flavor-specific behavior (constructors, `compute_partials!`,
# `compute_jacvec_product!`, accessors) dispatches on it.

"""
    DerivativeFlavor

Abstract supertype for the derivative computation strategy of an
[`ADExplicitComp`](@ref).

Concrete subtypes: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref),
[`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref),
[`MatrixFreeForwardReverseFlavor`](@ref).
"""
abstract type DerivativeFlavor end

"""
    AssembledFlavor <: DerivativeFlavor

Abstract supertype for derivative flavors that assemble and store a Jacobian
matrix.

Concrete subtypes: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref).
"""
abstract type AssembledFlavor <: DerivativeFlavor end

"""
    DenseFlavor <: AssembledFlavor

Derivative flavor that computes and stores the full dense Jacobian matrix via
`DifferentiationInterface.jacobian!`.
"""
struct DenseFlavor <: AssembledFlavor end

"""
    MatrixFreeFlavor <: DerivativeFlavor

Abstract supertype for matrix-free derivative flavors that compute
Jacobian-vector products without materializing the Jacobian.

Concrete subtypes: [`MatrixFreeForwardFlavor`](@ref),
[`MatrixFreeReverseFlavor`](@ref), [`MatrixFreeForwardReverseFlavor`](@ref).
"""
abstract type MatrixFreeFlavor <: DerivativeFlavor end

"""
    MatrixFreeForwardFlavor <: MatrixFreeFlavor

Derivative flavor that computes Jacobian-vector products (JVPs) via
`DifferentiationInterface.pushforward!`. Only `compute_jacvec_product!` with
`mode="fwd"` is supported.
"""
struct MatrixFreeForwardFlavor <: MatrixFreeFlavor end

"""
    MatrixFreeReverseFlavor <: MatrixFreeFlavor

Derivative flavor that computes vector-Jacobian products (VJPs) via
`DifferentiationInterface.pullback!`. Only `compute_jacvec_product!` with
`mode="rev"` is supported.
"""
struct MatrixFreeReverseFlavor <: MatrixFreeFlavor end

"""
    MatrixFreeForwardReverseFlavor <: MatrixFreeFlavor

Derivative flavor that computes Jacobian-vector products (JVPs) via
`DifferentiationInterface.pushforward!` with a forward-mode AD backend, *and*
vector-Jacobian products (VJPs) via `DifferentiationInterface.pullback!` with a
(reverse-mode) AD backend. Both `compute_jacvec_product!`/`apply_linear!`
directions (`mode="fwd"` and `mode="rev"`) are supported, using two separate
backends that are provided at construction time.
"""
struct MatrixFreeForwardReverseFlavor <: MatrixFreeFlavor end

"""
    SparseFlavor <: AssembledFlavor

Derivative flavor that computes and stores the sparse Jacobian using graph
coloring, via `DifferentiationInterface.jacobian!` with an `ADTypes.AutoSparse`
backend.

The `SparseFlavor` *type* and the `SparseDerivPrep` prep struct are declared
here in the main package, but the sparse *functionality* (constructors,
`compute_partials!`, sparsity detection, etc.) is provided by the
`OpenMDAOCoreSparseMatrixColoringsExt` extension, which loads when both
`SparseArrays` and `SparseMatrixColorings` are available.
"""
struct SparseFlavor <: AssembledFlavor end

# ---------------------------------------------------------------------------
# Flavor-specific derivative-prep structs (the `deriv_prep` field of
# `ADExplicitComp`). Only the *types* live here; their construction and the
# methods that read them are defined alongside the flavor-specific code.
# ---------------------------------------------------------------------------

"""
    DenseDerivPrep{TJ, TPrep}

Derivative preparation data for [`DenseFlavor`](@ref) components.

# Fields
* `J_ca::TJ`: dense `ComponentMatrix` storing the Jacobian of outputs w.r.t. inputs
* `prep::TPrep`: `DifferentiationInterface.JacobianPrep`, or `nothing` if prep was skipped
* `partial_copy_prep::DensePartialCopyPrep`: prepared metadata for scattering the
  sub-Jacobians of `J_ca` into the string-tuple-keyed partials dictionary
"""
struct DenseDerivPrep{TJ,TPrep}
    J_ca::TJ
    prep::TPrep
    partial_copy_prep::DensePartialCopyPrep
end

# Convenience constructor: builds the dense partials copy prep from `J_ca`.
# For the skip variant (`J_ca === nothing`), an empty prep is stored.
function DenseDerivPrep(J_ca, prep)
    if J_ca === nothing
        partial_copy_prep = DensePartialCopyPrep(String[], String[], UnitRange{Int}[], UnitRange{Int}[])
    else
        partial_copy_prep = DensePartialCopyPrep(J_ca)
    end
    return DenseDerivPrep(J_ca, prep, partial_copy_prep)
end

"""
    MatrixFreeDerivPrep{TdX, TdY, TPrep}

Derivative preparation data for [`MatrixFreeForwardFlavor`](@ref) and
[`MatrixFreeReverseFlavor`](@ref) components.

# Fields
* `dX_ca::TdX`: input tangent/cotangent `ComponentVector` buffer
* `dY_ca::TdY`: output tangent/cotangent `ComponentVector` buffer
* `prep::TPrep`: `DifferentiationInterface.PushforwardPrep` (for
  [`MatrixFreeForwardFlavor`](@ref)) or `PullbackPrep` (for
  [`MatrixFreeReverseFlavor`](@ref)), or their `No*Prep` skip-variants
"""
struct MatrixFreeDerivPrep{TdX,TdY,TPrep}
    dX_ca::TdX
    dY_ca::TdY
    prep::TPrep
end

"""
    MatrixFreeForwardReverseDerivPrep{TdX, TdY, TPrepFwd, TPrepRev}

Derivative preparation data for [`MatrixFreeForwardReverseFlavor`](@ref)
components, which use separate forward- and reverse-mode AD backends.

# Fields
* `dX_ca::TdX`: input `ComponentVector` buffer — the tangent (seed) for
  forward-mode JVPs and the cotangent (result) for reverse-mode VJPs
* `dY_ca::TdY`: output `ComponentVector` buffer — the tangent (result) for
  forward-mode JVPs and the cotangent (seed) for reverse-mode VJPs
* `prep_fwd::TPrepFwd`: `DifferentiationInterface.PushforwardPrep` created with
  the forward backend (or `nothing` if prep was skipped)
* `prep_rev::TPrepRev`: `DifferentiationInterface.PullbackPrep` created with the
  reverse backend (or `nothing` if prep was skipped)
"""
struct MatrixFreeForwardReverseDerivPrep{TdX,TdY,TPrepFwd,TPrepRev}
    dX_ca::TdX
    dY_ca::TdY
    prep_fwd::TPrepFwd
    prep_rev::TPrepRev
end

"""
    SparseDerivPrep{TJ, TPrep}

Derivative preparation data for [`SparseFlavor`](@ref) components.

# Fields
* `J_ca_sparse::TJ`: sparse `ComponentMatrix` (SparseMatrixCSC-backed) storing the
  Jacobian sparsity pattern and values
* `prep::TPrep`: `DifferentiationInterface.JacobianPrep` with graph coloring
  information
* `rcdict`: `Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}` mapping
  each `(output_name, input_name)` sub-Jacobian to its nonzero row and column
  indices (1-based)
"""
struct SparsePartialBlock
    output_name::String
    input_name::String
    rows::Vector{Int}
    cols::Vector{Int}
end

"""
    SparseDerivPrep{TJ, TPrep}

Derivative preparation data for [`SparseFlavor`](@ref) components.

# Fields
* `J_ca_sparse::TJ`: sparse `ComponentMatrix` (SparseMatrixCSC-backed) storing the
  Jacobian sparsity pattern and values
* `prep::TPrep`: `DifferentiationInterface.JacobianPrep` with graph coloring
  information
* `rcdict`: `Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}` mapping
  each `(output_name, input_name)` sub-Jacobian to its nonzero row and column
  indices (1-based)
* `blocks`: `Vector{SparsePartialBlock}` with the prepared global row/column
  indices used to scatter the sparse sub-Jacobians into the partials dictionary
"""
struct SparseDerivPrep{TJ,TPrep}
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}
    blocks::Vector{SparsePartialBlock}
end

# Convenience constructor: builds the sparse partials blocks from `J_ca_sparse`
# and `rcdict`. For the skip variant (`J_ca_sparse === nothing`), an empty block
# vector is stored.
function SparseDerivPrep(J_ca_sparse, prep, rcdict)
    if J_ca_sparse === nothing
        blocks = SparsePartialBlock[]
    else
        blocks = _prepare_sparse_blocks(J_ca_sparse, rcdict)
    end
    return SparseDerivPrep(J_ca_sparse, prep, rcdict, blocks)
end



# ---------------------------------------------------------------------------
# The unified explicit AD component.
# ---------------------------------------------------------------------------

"""
    ADExplicitComp{F<:DerivativeFlavor, InPlace, TAD, TADRev, TFunc, TParams, TCompute, TX, TY, TDP, TXCS, TYCS}
        <: AbstractExplicitComp

An explicit AD component parameterized by derivative flavor `F`.

# Type parameters
* `F<:DerivativeFlavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref),
  [`MatrixFreeReverseFlavor`](@ref), [`MatrixFreeForwardReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `InPlace::Bool`: `true` for in-place functions `f!(Y, X, params)`, `false` for
  out-of-place `Y = f(X, params)`
* `TAD`: the AD backend type (`<:ADTypes.AbstractADType`) used for
  forward-mode (JVP) derivatives
* `TADRev`: the AD backend type used for reverse-mode (VJP) derivatives;
  `Nothing` unless `F` is [`MatrixFreeForwardReverseFlavor`](@ref)

# Fields
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation backend
* `ad_backend_rev`: second AD backend for reverse-mode derivatives (`nothing`
  unless the flavor is [`MatrixFreeForwardReverseFlavor`](@ref))
* `func`: the user-defined function (`f!` for in-place, `f` for out-of-place)
* `params`: the parameters passed to `func` (captured by `compute_adable`)
* `compute_adable`: closure compatible with DifferentiationInterface.jl, built from `func` and `params`
* `X_ca`: `ComponentVector` of inputs (`Float64`)
* `Y_ca`: `ComponentVector` of outputs (`Float64`); `nothing` for out-of-place
* `deriv_prep`: flavor-specific prep data ([`DenseDerivPrep`](@ref),
  [`MatrixFreeDerivPrep`](@ref), or [`SparseDerivPrep`](@ref))
* `units_dict`: `Dict{Symbol,String}` mapping variable names to OpenMDAO units
* `tags_dict`: `Dict{Symbol,Vector{String}}` mapping variable names to tags
* `shape_by_conn_dict`: `Dict{Symbol,Bool}` for connection-determined shapes
* `copy_shape_dict`: `Dict{Symbol,Symbol}` for shape-copying
* `X_ca_cs`: `ComplexF64` copy of `X_ca` (for Python-side complex-step)
* `Y_ca_cs`: `ComplexF64` copy of `Y_ca` (for Python-side complex-step); `nothing` for out-of-place
* `input_copy_prep::InputCopyPrep`: prepared metadata for copying the `inputs` dict into `X_ca`
* `output_copy_prep::InputCopyPrep`: prepared metadata for copying `Y_ca` into the `outputs` dict
"""
struct ADExplicitComp{F<:DerivativeFlavor, InPlace, TAD, TADRev, TFunc, TParams, TCompute, TX, TY, TDP, TXCS, TYCS} <: AbstractExplicitComp
    ad_backend::TAD
    ad_backend_rev::TADRev
    func::TFunc
    params::TParams
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY            # nothing for out-of-place
    deriv_prep::TDP
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS       # nothing for out-of-place
    input_copy_prep::InputCopyPrep
    output_copy_prep::InputCopyPrep

    function ADExplicitComp{F, InPlace}(
            ad_backend, func, params, compute_adable, X_ca, Y_ca, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict,
            X_ca_cs, Y_ca_cs, input_copy_prep, output_copy_prep;
            ad_backend_rev = nothing
        ) where {F<:DerivativeFlavor, InPlace}
        return new{F, InPlace,
                   typeof(ad_backend), typeof(ad_backend_rev), typeof(func), typeof(params), typeof(compute_adable),
                   typeof(X_ca), typeof(Y_ca),
                   typeof(deriv_prep),
                   typeof(X_ca_cs), typeof(Y_ca_cs)}(
            ad_backend, ad_backend_rev, func, params, compute_adable, X_ca, Y_ca, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict,
            X_ca_cs, Y_ca_cs, input_copy_prep, output_copy_prep)
    end
end

# Build the DifferentiationInterface-compatible closure for an explicit AD
# component from the user-defined function and parameters.
function _make_compute_adable(::Val{true}, func, params)
    return let func=func, params=params
        (Y, X) -> begin
            func(Y, X, params)
            return nothing
        end
    end
end

function _make_compute_adable(::Val{false}, func, params)
    return let func=func, params=params
        (X,) -> begin
            return func(X, params)
        end
    end
end

# ---------------------------------------------------------------------------
# Shared accessors (work for any flavor — read only common fields).
# ---------------------------------------------------------------------------

get_callback(comp::ADExplicitComp) = comp.compute_adable
get_func(comp::ADExplicitComp) = comp.func
get_params(comp::ADExplicitComp) = comp.params
get_backend(comp::ADExplicitComp) = comp.ad_backend
get_prep(comp::ADExplicitComp) = comp.deriv_prep.prep

"""
    get_backend_fwd(comp) -> backend
    get_backend_rev(comp) -> backend

Return the AD backend used for forward-mode (JVP) / reverse-mode (VJP)
derivative computations, respectively. For all flavors except
[`MatrixFreeForwardReverseFlavor`](@ref), `get_backend_fwd` returns the comp's
only backend and `get_backend_rev` returns `nothing`.
"""
get_backend_fwd(comp::ADExplicitComp) = comp.ad_backend
get_backend_rev(comp::ADExplicitComp) = comp.ad_backend_rev

get_input_ca(::Type{Float64}, comp::ADExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::ADExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::ADExplicitComp) = comp.X_ca
get_input_ca(comp::ADExplicitComp) = get_input_ca(Float64, comp)

get_output_ca(::Type{Float64}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::ADExplicitComp{<:DerivativeFlavor, true}) = comp.Y_ca

get_output_ca(::Type{Float64}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(Float64, comp))
get_output_ca(::Type{ComplexF64}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(ComplexF64, comp))
get_output_ca(::Type{Any}, comp::ADExplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_input_ca(Float64, comp))

get_output_ca(comp::ADExplicitComp) = get_output_ca(Float64, comp)

get_units(comp::ADExplicitComp, varname) = get(comp.units_dict, varname, "unitless")
get_tags(comp::ADExplicitComp, varname) = get(comp.tags_dict, varname, Vector{String}())

# Return the `copy_shape` target for `varname` as a `String` (or `nothing` if unset).
# `copy_shape_dict` maps a variable's `Symbol` key to another `Symbol` key whose shape
# should be copied; `VarData.copy_shape` expects a `String`.
function get_copy_shape(comp::ADExplicitComp, varname)
    cs = get(comp.copy_shape_dict, varname, nothing)
    return cs === nothing ? nothing : string(cs)
end

function get_input_var_data(self::ADExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::ADExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

function OpenMDAOCore.setup(self::ADExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)

    return input_data, output_data, Vector{PartialsData}()
end

function OpenMDAOCore.compute!(self::ADExplicitComp{<:DerivativeFlavor, true}, inputs, outputs)
    # I used to try to do eltype(valtype(inputs)) but that would return `Any` if there were arrays and scalars in `inputs`.
    TF = eltype(inputs[first(keys(inputs))])

    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(TF, self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    # Call the actual function.
    Y_ca = get_output_ca(TF, self)
    f! = get_callback(self)
    f!(Y_ca, X_ca)

    # Copy the output `ComponentArray` to the outputs.
    _copy_component_outputs!(getdata(Y_ca), outputs, self.output_copy_prep)

    return nothing
end

function OpenMDAOCore.compute!(self::ADExplicitComp{<:DerivativeFlavor, false}, inputs, outputs)
    # I used to try to do TF = eltype(valtype(inputs)) but that would return `Any` if there were arrays and scalars in `inputs`.
    TF = eltype(inputs[first(keys(inputs))])
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(TF, self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    # Call the actual function.
    f = get_callback(self)
    Y_ca = f(X_ca)

    # Copy the output `ComponentArray` to the outputs.
    _copy_component_outputs!(getdata(Y_ca), outputs, self.output_copy_prep)

    return nothing
end

# ---------------------------------------------------------------------------
# Flavor-dispatch accessors (trivial field reads).
# ---------------------------------------------------------------------------

# Dense: read the dense Jacobian ComponentMatrix.
get_jacobian_ca(comp::ADExplicitComp{DenseFlavor}) = comp.deriv_prep.J_ca

# Sparse: read the sparse Jacobian ComponentMatrix. (The sparse *methods* live
# in the extension, but this accessor is a trivial field read and is fine here.)
get_jacobian_ca(comp::ADExplicitComp{SparseFlavor}) = comp.deriv_prep.J_ca_sparse

# Matrix-free: read the tangent/cotangent buffers.
get_dinput_ca(comp::ADExplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dX_ca
get_doutput_ca(comp::ADExplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dY_ca

# Sparse: read the row/col index dict.
get_rows_cols_dict(comp::ADExplicitComp{SparseFlavor}) = comp.deriv_prep.rcdict

# ---------------------------------------------------------------------------
# `has_*` introspection — flavor-dispatched.
# ---------------------------------------------------------------------------

# Dense and Sparse: support setup_partials + compute_partials, not jacvec.
has_setup_partials(self::ADExplicitComp{DenseFlavor}) = true
has_compute_partials(self::ADExplicitComp{DenseFlavor}) = true
has_compute_jacvec_product(self::ADExplicitComp{DenseFlavor}) = false

has_setup_partials(self::ADExplicitComp{SparseFlavor}) = true
has_compute_partials(self::ADExplicitComp{SparseFlavor}) = true
has_compute_jacvec_product(self::ADExplicitComp{SparseFlavor}) = false

# Matrix-free: support setup_partials + jacvec, not compute_partials.
has_setup_partials(self::ADExplicitComp{<:MatrixFreeFlavor}) = true
has_compute_partials(self::ADExplicitComp{<:MatrixFreeFlavor}) = false
has_compute_jacvec_product(self::ADExplicitComp{<:MatrixFreeFlavor}) = true

# ---------------------------------------------------------------------------
# ADImplicitComp type definition
# ---------------------------------------------------------------------------

"""
    ADImplicitComp{F<:DerivativeFlavor,InPlace,TAD,TADRev,TCompute,TYX,TR,TYXCS,TRCS,TDP,TYR,TXR,TYAx,TXAx} <: AbstractImplicitComp

An `<:AbstractImplicitComp` differentiated via automatic differentiation

The full `[dR/dY | dR/dX]` Jacobian is computed in a single `DifferentiationInterface.jacobian!`
call by concatenating the output (state) vector `Y` and input vector `X` into a combined vector `YX` and
differentiating with respect to `YX`.

* `F<:DerivativeFlavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref),
  [`MatrixFreeReverseFlavor`](@ref), [`MatrixFreeForwardReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `InPlace::Bool`: `true` for in-place functions `f!(R, Y, X, params)`, `false` for
  out-of-place `R = f(Y, X, params)`
* `TAD`: the AD backend type (`<:ADTypes.AbstractADType`) used for
  forward-mode (JVP) derivatives
* `TADRev`: the AD backend type used for reverse-mode (VJP) derivatives;
  `Nothing` unless `F` is [`MatrixFreeForwardReverseFlavor`](@ref)

# Fields
* `ad_backend::TAD`: `<:ADTypes.AbstractADType` automatic differentiation backend
* `ad_backend_rev::TADRev`: second AD backend for reverse-mode derivatives
  (`nothing` unless the flavor is [`MatrixFreeForwardReverseFlavor`](@ref))
* `compute_adable::TCompute`: closure of the form `compute_adable(R, YX)` (in-place) or
  `R = compute_adable(YX)` (out-of-place) compatible with DifferentiationInterface.jl,
where `YX` is the concatenated output (state)+input `ComponentVector` and `R` is the residuals
  `ComponentVector`. Built from `func` and `params`.
* `func::TFunc`: the user-defined function (`f!` for in-place, `f` for out-of-place)
* `params::TParams`: the parameters passed to `func`
* `guess_func::TGuessFunc`: user-defined `guess_nonlinear` callback function (`nothing` if not provided; see [`guess_nonlinear!`](@ref))
* `YX_ca::TYX`: concatenated `ComponentVector` of outputs (state) followed by inputs (the differentiation variable)
* `R_ca::TR`: `ComponentVector` of residuals
* `YX_ca_cs::TYXCS`: `ComplexF64` version of `YX_ca` (for the complex-step method)
* `R_ca_cs::TRCS`: `ComplexF64` version of `R_ca` (for the complex-step method)
* `deriv_prep::TDP`: flavor-specific derivative preparation data
* `units_dict::Dict{Symbol,String}`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict::Dict{Symbol,Vector{String}}`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict::Dict{Symbol,Bool}`: `Dict` indicating if a variable's shape is determined by a connection
* `copy_shape_dict::Dict{Symbol,Symbol}`: `Dict` mapping variable names to the names of variables whose shapes they copy
* `Y_range::TYR`: range of indices for outputs (states) within `YX_ca`
* `X_range::TXR`: range of indices for inputs within `YX_ca`
* `Y_axes::TYAx`: axes information for outputs (states)
* `X_axes::TXAx`: axes information for inputs
* `state_copy_prep::InputCopyPrep`: prepared metadata for copying the `outputs` dict into (or out of) the states part of `YX_ca`
* `input_copy_prep::InputCopyPrep`: prepared metadata for copying the `inputs` dict into (or out of) the inputs part of `YX_ca`
* `residual_copy_prep::InputCopyPrep`: prepared metadata for copying the `residuals` dict into (or out of) `R_ca`
"""
struct ADImplicitComp{F<:DerivativeFlavor,InPlace,TAD,TADRev,TFunc,TParams,TGuessFunc,TCompute,TYX,TR,TYXCS,TRCS,TDP,TUD,TTD,TSBC,TCS,TYR,TXR,TYAx,TXAx} <: AbstractImplicitComp
    ad_backend::TAD
    ad_backend_rev::TADRev
    compute_adable::TCompute
    func::TFunc
    params::TParams
    guess_func::TGuessFunc
    YX_ca::TYX
    R_ca::TR
    YX_ca_cs::TYXCS
    R_ca_cs::TRCS
    deriv_prep::TDP
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    Y_range::TYR
    X_range::TXR
    Y_axes::TYAx
    X_axes::TXAx
    state_copy_prep::InputCopyPrep
    input_copy_prep::InputCopyPrep
    residual_copy_prep::InputCopyPrep

    function ADImplicitComp{F, InPlace}(ad_backend, compute_adable, func, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, state_copy_prep, input_copy_prep, residual_copy_prep; ad_backend_rev = nothing, guess_func = nothing) where {F<:DerivativeFlavor, InPlace}
        return new{F, InPlace,
                   typeof(ad_backend), typeof(ad_backend_rev), typeof(func), typeof(params), typeof(guess_func), typeof(compute_adable),
                   typeof(YX_ca), typeof(R_ca),
                   typeof(YX_ca_cs), typeof(R_ca_cs),
                   typeof(deriv_prep),
                   typeof(units_dict), typeof(tags_dict),
                   typeof(shape_by_conn_dict), typeof(copy_shape_dict),
                   typeof(Y_range), typeof(X_range), typeof(Y_axes), typeof(X_axes)}(
            ad_backend, ad_backend_rev, compute_adable, func, params, guess_func, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, state_copy_prep, input_copy_prep, residual_copy_prep)
    end
end

# Build the DifferentiationInterface-compatible closure for an implicit AD
# component from the user-defined function, parameters, and the YX geometry.
function _make_implicit_compute_adable(inplace::Bool, func, params, Y_range, X_range, Y_axes, X_axes)
    if inplace
        return let func=func, params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
            (R, YX) -> begin
                Y = ComponentArray(@view(YX[Y_range]), Y_axes)
                X = ComponentArray(@view(YX[X_range]), X_axes)
                func(R, Y, X, params)
                return nothing
            end
        end
    else
        return let func=func, params=params, Y_range=Y_range, X_range=X_range, Y_axes=Y_axes, X_axes=X_axes
            YX -> func(ComponentArray(@view(YX[Y_range]), Y_axes), ComponentArray(@view(YX[X_range]), X_axes), params)
        end
    end
end

# ---------------------------------------------------------------------------
# ADImplicitComp accessors
# ---------------------------------------------------------------------------

get_callback(comp::ADImplicitComp) = comp.compute_adable
get_func(comp::ADImplicitComp) = comp.func
get_params(comp::ADImplicitComp) = comp.params
get_backend(comp::ADImplicitComp) = comp.ad_backend
get_prep(comp::ADImplicitComp) = comp.deriv_prep.prep

"""
    get_backend_fwd(comp) -> backend
    get_backend_rev(comp) -> backend

Return the AD backend used for forward-mode (JVP) / reverse-mode (VJP)
derivative computations, respectively. For all flavors except
[`MatrixFreeForwardReverseFlavor`](@ref), `get_backend_fwd` returns the comp's
only backend and `get_backend_rev` returns `nothing`.
"""
get_backend_fwd(comp::ADImplicitComp) = comp.ad_backend
get_backend_rev(comp::ADImplicitComp) = comp.ad_backend_rev

get_combined_ca(::Type{<:AbstractFloat}, comp::ADImplicitComp) = comp.YX_ca
get_combined_ca(::Type{Complex{T}}, comp::ADImplicitComp) where {T<:AbstractFloat} = comp.YX_ca_cs
get_combined_ca(comp::ADImplicitComp) = get_combined_ca(AbstractFloat, comp)

get_residual_ca(::Type{<:AbstractFloat}, comp::ADImplicitComp{<:DerivativeFlavor, true}) = comp.R_ca
get_residual_ca(::Type{Complex{T}}, comp::ADImplicitComp{<:DerivativeFlavor, true}) where {T<:AbstractFloat} = comp.R_ca_cs
get_residual_ca(::Type{<:AbstractFloat}, comp::ADImplicitComp{<:DerivativeFlavor, false}) = get_callback(comp)(get_combined_ca(AbstractFloat, comp))
get_residual_ca(TF::Type{Complex{T}}, comp::ADImplicitComp{<:DerivativeFlavor, false}) where {T<:AbstractFloat} = get_callback(comp)(get_combined_ca(TF, comp))
get_residual_ca(comp::ADImplicitComp) = get_residual_ca(AbstractFloat, comp)

# Dense: read the dense Jacobian ComponentMatrix.
get_jacobian_ca(comp::ADImplicitComp{DenseFlavor}) = comp.deriv_prep.J_ca

# Matrix-free: read the tangent/cotangent buffers.
get_dcombined_ca(comp::ADImplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dX_ca
get_dresidual_ca(comp::ADImplicitComp{<:MatrixFreeFlavor}) = comp.deriv_prep.dY_ca

input_keys(comp::ADImplicitComp) = keys(indexmap(only(comp.X_axes)))
output_keys(comp::ADImplicitComp) = keys(indexmap(only(comp.Y_axes)))

function get_input_ca(::Type{TF}, comp::ADImplicitComp) where {TF}
    YX = get_combined_ca(TF, comp)
    return ComponentArray(@view(YX[comp.X_range]), comp.X_axes)
end
get_input_ca(comp::ADImplicitComp) = get_input_ca(AbstractFloat, comp)

function get_output_ca(::Type{TF}, comp::ADImplicitComp) where {TF}
    YX = get_combined_ca(TF, comp)
    return ComponentArray(@view(YX[comp.Y_range]), comp.Y_axes)
end
get_output_ca(comp::ADImplicitComp) = get_output_ca(AbstractFloat, comp)

function get_dinput_ca(comp::ADImplicitComp{<:MatrixFreeFlavor})
    dYX = get_dcombined_ca(comp)
    return ComponentArray(@view(dYX[comp.X_range]), comp.X_axes)
end

function get_doutput_ca(comp::ADImplicitComp{<:MatrixFreeFlavor})
    dYX = get_dcombined_ca(comp)
    return ComponentArray(@view(dYX[comp.Y_range]), comp.Y_axes)
end

# ---------------------------------------------------------------------------
# OpenMDAO `setup` machinery for `ADImplicitComp` (all flavors).
# ---------------------------------------------------------------------------

function get_units(comp::ADImplicitComp, varname)
    return get(comp.units_dict, varname, "unitless")
end

function get_tags(comp::ADImplicitComp, varname)
    return get(comp.tags_dict, varname, Vector{String}())
end

function get_copy_shape(comp::ADImplicitComp, varname)
    cs = get(comp.copy_shape_dict, varname, nothing)
    return cs === nothing ? nothing : string(cs)
end

function get_input_var_data(self::ADImplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::ADImplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_copy_shape(self, k)) for k in keys(ca)]
end

# Dense implicit components declare a single dense ("*", "*") partial.
function get_partials_data(self::ADImplicitComp{DenseFlavor})
    return [PartialsData("*", "*")]
end

# Matrix-free implicit components declare no partials; OpenMDAO discovers the
# matrix-free API through the `apply_linear` method override in `omjlcomps`.
function get_partials_data(self::ADImplicitComp{<:MatrixFreeFlavor})
    return Vector{PartialsData}()
end

function OpenMDAOCore.setup(self::ADImplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)

    return input_data, output_data, Vector{PartialsData}()
end

function OpenMDAOCore.setup_partials(self::ADImplicitComp, input_sizes, output_sizes)
    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = _update_prep(self, input_sizes_ca, output_sizes_ca)

    return self_new, get_partials_data(self_new)
end

# ---------------------------------------------------------------------------
# ADImplicitComp has_* methods
# ---------------------------------------------------------------------------

# Dense: support apply_nonlinear + linearize, not apply_linear (matrix-free).
has_apply_nonlinear(self::ADImplicitComp{DenseFlavor}) = true
has_linearize(self::ADImplicitComp{DenseFlavor}) = true
has_apply_linear(self::ADImplicitComp{DenseFlavor}) = false

# Sparse: support apply_nonlinear + linearize, not apply_linear (matrix-free).
# The `ADImplicitComp{SparseFlavor}` constructors and the `linearize!` methods
# are provided by the `OpenMDAOCoreSparseMatrixColoringsExt` extension.
has_apply_nonlinear(self::ADImplicitComp{SparseFlavor}) = true
has_linearize(self::ADImplicitComp{SparseFlavor}) = true
has_apply_linear(self::ADImplicitComp{SparseFlavor}) = false

# All implicit AD flavors support `setup_partials` (used to rebuild the
# component when `shape_by_conn`/`copy_shape` variables are resized, and to
# declare partials after the shapes are known).
has_setup_partials(self::ADImplicitComp{<:DerivativeFlavor}) = true

# Matrix-free: support apply_nonlinear + apply_linear, not linearize (assembled).
has_apply_nonlinear(self::ADImplicitComp{<:MatrixFreeFlavor}) = true
has_linearize(self::ADImplicitComp{<:MatrixFreeFlavor}) = false
has_apply_linear(self::ADImplicitComp{<:MatrixFreeFlavor}) = true

# ---------------------------------------------------------------------------
# `apply_nonlinear!` — shared by all implicit AD flavors.
#
# The residual is computed by calling the `compute_adable` closure directly;
# this is the same for every flavor, since only the *derivative* computation
# differs between them.
# ---------------------------------------------------------------------------

function apply_nonlinear!(comp::ADImplicitComp{<:DerivativeFlavor, true}, inputs, outputs, residuals)
    # Note: `eltype(valtype(inputs))` returns `Any` for the `Dict{String, Any}` that
    # omjlcomps passes in, so use the first input's element type instead (mirroring
    # `compute!` for the explicit comps).
    TF = eltype(inputs[first(keys(inputs))])
    YX_ca = get_combined_ca(TF, comp)
    R_ca = get_residual_ca(TF, comp)

    # Copy outputs (states) and inputs into the combined YX vector
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Call the compute_adable closure directly.
    f! = get_callback(comp)
    f!(R_ca, YX_ca)

    # Scatter R_ca back to the residuals dict.
    _copy_component_outputs!(getdata(R_ca), residuals, comp.residual_copy_prep)

    return nothing
end

# ---------------------------------------------------------------------------
# `guess_nonlinear!` — optional, only for comps that provide `guess_func`.
# ---------------------------------------------------------------------------

"""
    guess_nonlinear!(comp::ADImplicitComp, inputs, outputs, residuals)

Called by OpenMDAO before a nonlinear solve, giving the component a chance to provide initial guesses for the states.
Only defined for [`ADImplicitComp`](@ref)s that were created with a `guess_func`; for those, `has_guess_nonlinear` returns `true`.

The in-place user function has the form `f_guess!(Y_ca, R_ca, X_ca, params)`: it receives the current states in the output `ComponentVector` `Y_ca` (which it overwrites with its state guesses), the current residuals in the residual `ComponentVector` `R_ca` (pre-filled from OpenMDAO, which guarantees they are fresh — computed from the current inputs and outputs), and the inputs `X_ca`. Note that OpenMDAO treats the residuals as read-only in `guess_nonlinear`, so `R_ca` cannot be used to update the residuals; it is provided for context only.

The out-of-place user function has the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`: it receives the current states, the fresh current residuals, and the inputs, and returns the updated states as a single `ComponentVector`. The residuals are read-only in `guess_nonlinear`, so no residual result is returned or used.
"""
function OpenMDAOCore.guess_nonlinear!(comp::ADImplicitComp{<:DerivativeFlavor, true}, inputs, outputs, residuals)
    TF = eltype(inputs[first(keys(inputs))])
    X_ca = get_input_ca(TF, comp)
    Y_ca = get_output_ca(TF, comp)
    R_ca = get_residual_ca(TF, comp)

    # Copy the inputs, outputs (states), and residuals into the ComponentArrays.
    _copy_component_inputs!(getdata(X_ca), inputs, comp.input_copy_prep)
    _copy_component_inputs!(getdata(Y_ca), outputs, comp.state_copy_prep)
    _copy_component_inputs!(getdata(R_ca), residuals, comp.residual_copy_prep)

    # Let the user update the guesses in place.
    f_guess! = comp.guess_func
    f_guess! === nothing && throw(OpenMDAOMethodError(comp, "guess_nonlinear!"))
    f_guess!(Y_ca, R_ca, X_ca, comp.params)

    # Scatter the guesses back to the outputs dict. Note that the residuals
    # are deliberately NOT written back: OpenMDAO makes the residual vector
    # read-only in `guess_nonlinear`.
    _copy_component_outputs!(getdata(Y_ca), outputs, comp.state_copy_prep)

    return nothing
end

function OpenMDAOCore.guess_nonlinear!(comp::ADImplicitComp{<:DerivativeFlavor, false}, inputs, outputs, residuals)
    TF = eltype(inputs[first(keys(inputs))])
    X_ca = get_input_ca(TF, comp)
    Y_ca = get_output_ca(TF, comp)
    R_ca = get_residual_ca(TF, comp)

    # Copy the inputs, outputs (states), and residuals into the ComponentArrays.
    _copy_component_inputs!(getdata(X_ca), inputs, comp.input_copy_prep)
    _copy_component_inputs!(getdata(Y_ca), outputs, comp.state_copy_prep)
    _copy_component_inputs!(getdata(R_ca), residuals, comp.residual_copy_prep)

    # Call the out-of-place guess function, which returns the updated states.
    # The residuals are read-only in `guess_nonlinear`, so the guess function
    # does not return any residual result.
    f_guess = comp.guess_func
    f_guess === nothing && throw(OpenMDAOMethodError(comp, "guess_nonlinear!"))
    Y_ca_new = f_guess(Y_ca, R_ca, X_ca, comp.params)

    # Scatter the guesses back to the outputs dict. Note that the residuals
    # are deliberately NOT written back: OpenMDAO makes the residual vector
    # read-only in `guess_nonlinear`.
    _copy_component_outputs!(getdata(Y_ca_new), outputs, comp.state_copy_prep)

    return nothing
end

# Value-based override: the type-based `which`-checking fallback would return
# `true` for every `ADImplicitComp` once `guess_nonlinear!` is defined, which
# would make omjlcomps override OpenMDAO's `guess_nonlinear` for comps without
# a `guess_func`.
has_guess_nonlinear(comp::ADImplicitComp) = comp.guess_func !== nothing

function apply_nonlinear!(comp::ADImplicitComp{<:DerivativeFlavor, false}, inputs, outputs, residuals)
    # Note: `eltype(valtype(inputs))` returns `Any` for the `Dict{String, Any}` that
    # omjlcomps passes in, so use the first input's element type instead (mirroring
    # `compute!` for the explicit comps).
    TF = eltype(inputs[first(keys(inputs))])
    YX_ca = get_combined_ca(TF, comp)

    # Copy outputs (states) and inputs into the combined YX vector
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Call the out-of-place closure and scatter the result.
    f = get_callback(comp)
    R_ca = f(YX_ca)

    # Scatter R_ca back to the residuals dict.
    _copy_component_outputs!(getdata(R_ca), residuals, comp.residual_copy_prep)

    return nothing
end
