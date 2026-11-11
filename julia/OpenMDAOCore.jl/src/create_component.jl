# Convenience constructors for `ADExplicitComp`, inspired by
# OpenMDAO4Core.jl's `create_explicit_component`.
#
# These wrap the flavor-specific `ADExplicitComp(flavor, ...)` constructors with
# an additional validation step: for matrix-free flavors, the AD backend must
# support the required derivative mode (JVP for `MatrixFreeForwardFlavor`, VJP
# for `MatrixFreeReverseFlavor`); otherwise an `ArgumentError` is thrown with a
# helpful message. Dense and Sparse flavors have no extra check (a non-sparse
# backend passed with `SparseFlavor` yields a `MethodError` from the delegate's
# `TAD<:ADTypes.AutoSparse` constraint, matching direct `ADExplicitComp` use).

"""
    create_explicit_component(flavor::DerivativeFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an in-place [`ADExplicitComp`](@ref) from a user-defined function and output and input `ComponentVector`s.

`ad_backend` is validated against the flavor: [`DenseFlavor`](@ref) requires a dense (non-`AutoSparse`) backend; [`SparseFlavor`](@ref) requires an `AutoSparse` backend; [`MatrixFreeForwardFlavor`](@ref) requires a backend that supports JVPs (pushforwards); and [`MatrixFreeReverseFlavor`](@ref) requires a backend that supports VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::DerivativeFlavor, ad_backend::TAD,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    _check_flavor_backend(flavor, ad_backend)
    return ADExplicitComp(flavor, ad_backend, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

"""
    create_explicit_component(flavor::DerivativeFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an out-of-place [`ADExplicitComp`](@ref) from a user-defined function and input `ComponentVector`.

`ad_backend` is validated against the flavor: [`DenseFlavor`](@ref) requires a dense (non-`AutoSparse`) backend; [`SparseFlavor`](@ref) requires an `AutoSparse` backend; [`MatrixFreeForwardFlavor`](@ref) requires a backend that supports JVPs (pushforwards); and [`MatrixFreeReverseFlavor`](@ref) requires a backend that supports VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), [`MatrixFreeReverseFlavor`](@ref), or [`SparseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the second argument to `f`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::DerivativeFlavor, ad_backend::TAD,
        f::Function, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}
    _check_flavor_backend(flavor, ad_backend)
    return ADExplicitComp(flavor, ad_backend, f, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

"""
    create_explicit_component(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an in-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADExplicitComp`](@ref)
from a user-defined function and output and input `ComponentVector`s. JVPs
(`mode="fwd"`) use `ad_backend_fwd`, which must support pushforwards; VJPs
(`mode="rev"`) use `ad_backend_rev`, which must support pullbacks.

# Positional Arguments
* `flavor`: [`MatrixFreeForwardReverseFlavor`](@ref)
* `ad_backend_fwd`: `<:ADTypes.AbstractADType` backend used for forward-mode (JVP) derivatives
* `ad_backend_rev`: `<:ADTypes.AbstractADType` backend used for reverse-mode (VJP) derivatives
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::MatrixFreeForwardReverseFlavor, ad_backend_fwd::TADFwd, ad_backend_rev::TADRev,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TADFwd<:ADTypes.AbstractADType, TADRev<:ADTypes.AbstractADType}
    can_jvp(ad_backend_fwd) || throw(ArgumentError("AD backend $(ad_backend_fwd) does not support JVPs (pushforwards), which are required for the forward direction of MatrixFreeForwardReverseFlavor"))
    can_vjp(ad_backend_rev) || throw(ArgumentError("AD backend $(ad_backend_rev) does not support VJPs (pullbacks), which are required for the reverse direction of MatrixFreeForwardReverseFlavor"))
    return ADExplicitComp(flavor, ad_backend_fwd, ad_backend_rev, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

"""
    create_explicit_component(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADExplicitComp`](@ref)
from a user-defined function and input `ComponentVector`. JVPs (`mode="fwd"`) use
`ad_backend_fwd`, which must support pushforwards; VJPs (`mode="rev"`) use
`ad_backend_rev`, which must support pullbacks.

# Positional Arguments
* `flavor`: [`MatrixFreeForwardReverseFlavor`](@ref)
* `ad_backend_fwd`: `<:ADTypes.AbstractADType` backend used for forward-mode (JVP) derivatives
* `ad_backend_rev`: `<:ADTypes.AbstractADType` backend used for reverse-mode (VJP) derivatives
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the second argument to `f`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
"""
function create_explicit_component(flavor::MatrixFreeForwardReverseFlavor, ad_backend_fwd::TADFwd, ad_backend_rev::TADRev,
        f::Function, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(),
        force_skip_prep=false) where {TADFwd<:ADTypes.AbstractADType, TADRev<:ADTypes.AbstractADType}
    can_jvp(ad_backend_fwd) || throw(ArgumentError("AD backend $(ad_backend_fwd) does not support JVPs (pushforwards), which are required for the forward direction of MatrixFreeForwardReverseFlavor"))
    can_vjp(ad_backend_rev) || throw(ArgumentError("AD backend $(ad_backend_rev) does not support VJPs (pullbacks), which are required for the reverse direction of MatrixFreeForwardReverseFlavor"))
    return ADExplicitComp(flavor, ad_backend_fwd, ad_backend_rev, f, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict,
        shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict,
        force_skip_prep=force_skip_prep)
end

# ---------------------------------------------------------------------------
# Convenience constructors for `ADImplicitComp`
# ---------------------------------------------------------------------------

# Validates that `ad_backend` is compatible with `flavor`, throwing an
# `ArgumentError` if not.
function _check_flavor_backend(flavor::DerivativeFlavor, ad_backend::ADTypes.AbstractADType)
    if flavor isa SparseFlavor
        ad_backend isa ADTypes.AutoSparse || throw(ArgumentError("SparseFlavor requires an AutoSparse backend (<:ADTypes.AutoSparse), so that the sparsity structure of the component can be detected and exploited, but got $(ad_backend)"))
    elseif flavor isa DenseFlavor
        !(ad_backend isa ADTypes.AutoSparse) || throw(ArgumentError("DenseFlavor requires a dense (non-AutoSparse) backend, but got $(ad_backend)"))
    elseif flavor isa MatrixFreeForwardFlavor
        can_jvp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support JVPs (pushforwards), which are required for MatrixFreeForwardFlavor"))
    elseif flavor isa MatrixFreeReverseFlavor
        can_vjp(ad_backend) || throw(ArgumentError("AD backend $(ad_backend) does not support VJPs (pullbacks), which are required for MatrixFreeReverseFlavor"))
    end
    return nothing
end

"""
    create_implicit_component(flavor::DerivativeFlavor, ::Val{true}, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false, guess_nonlinear=nothing)

Create an in-place [`ADImplicitComp`](@ref) from a user-defined function and state/output and input `ComponentVector`s.

`ad_backend` is validated against the flavor: [`DenseFlavor`](@ref) requires a dense (non-`AutoSparse`) backend; [`SparseFlavor`](@ref) requires an `AutoSparse` backend; [`MatrixFreeForwardFlavor`](@ref) requires a backend that supports JVPs (pushforwards); and [`MatrixFreeReverseFlavor`](@ref) requires a backend that supports VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), or [`MatrixFreeReverseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f!`: function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the fourth argument to `f!`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::DerivativeFlavor, ::Val{true}, ad_backend::TAD,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TAD<:ADTypes.AbstractADType}
    _check_flavor_backend(flavor, ad_backend)
    return ADImplicitComp(flavor, Val(true), ad_backend, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_func=guess_nonlinear)
end

"""
    create_implicit_component(flavor::DerivativeFlavor, ::Val{false}, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false, guess_nonlinear=nothing)

Create an out-of-place [`ADImplicitComp`](@ref) from a user-defined function and state/output and input `ComponentVector`s.

`ad_backend` is validated against the flavor: [`DenseFlavor`](@ref) requires a dense (non-`AutoSparse`) backend; [`SparseFlavor`](@ref) requires an `AutoSparse` backend; [`MatrixFreeForwardFlavor`](@ref) requires a backend that supports JVPs (pushforwards); and [`MatrixFreeReverseFlavor`](@ref) requires a backend that supports VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), or [`MatrixFreeReverseFlavor`](@ref)
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f`: function of the form `R_ca = f(Y_ca, X_ca, params)` which returns residuals `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::DerivativeFlavor, ::Val{false}, ad_backend::TAD,
        f::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TAD<:ADTypes.AbstractADType}
    _check_flavor_backend(flavor, ad_backend)
    return ADImplicitComp(flavor, Val(false), ad_backend, f, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_func=guess_nonlinear)
end

"""
    create_implicit_component(::MatrixFreeForwardReverseFlavor, ::Val{true}, ad_backend_fwd, ad_backend_rev, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing)

Create an in-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADImplicitComp`](@ref)
from a user-defined function and state/output and input `ComponentVector`s. JVPs
(`mode="fwd"`) use `ad_backend_fwd`, which must support pushforwards; VJPs
(`mode="rev"`) use `ad_backend_rev`, which must support pullbacks.

# Positional Arguments
* `flavor`: [`MatrixFreeForwardReverseFlavor`](@ref)
* `ad_backend_fwd`: `<:ADTypes.AbstractADType` backend used for forward-mode (JVP) derivatives
* `ad_backend_rev`: `<:ADTypes.AbstractADType` backend used for reverse-mode (VJP) derivatives
* `f!`: function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the fourth argument to `f!`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::MatrixFreeForwardReverseFlavor, ::Val{true}, ad_backend_fwd::TADFwd, ad_backend_rev::TADRev,
        f!::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TADFwd<:ADTypes.AbstractADType, TADRev<:ADTypes.AbstractADType}
    can_jvp(ad_backend_fwd) || throw(ArgumentError("AD backend $(ad_backend_fwd) does not support JVPs (pushforwards), which are required for the forward direction of MatrixFreeForwardReverseFlavor"))
    can_vjp(ad_backend_rev) || throw(ArgumentError("AD backend $(ad_backend_rev) does not support VJPs (pullbacks), which are required for the reverse direction of MatrixFreeForwardReverseFlavor"))
    return ADImplicitComp(flavor, Val(true), ad_backend_fwd, ad_backend_rev, f!, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_func=guess_nonlinear)
end

"""
    create_implicit_component(::MatrixFreeForwardReverseFlavor, ::Val{false}, ad_backend_fwd, ad_backend_rev, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing)

Create an out-of-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADImplicitComp`](@ref)
from a user-defined function and state/output and input `ComponentVector`s. JVPs
(`mode="fwd"`) use `ad_backend_fwd`, which must support pushforwards; VJPs
(`mode="rev"`) use `ad_backend_rev`, which must support pullbacks.

# Positional Arguments
* `flavor`: [`MatrixFreeForwardReverseFlavor`](@ref)
* `ad_backend_fwd`: `<:ADTypes.AbstractADType` backend used for forward-mode (JVP) derivatives
* `ad_backend_rev`: `<:ADTypes.AbstractADType` backend used for reverse-mode (VJP) derivatives
* `f`: function of the form `R_ca = f(Y_ca, X_ca, params)` which returns residuals `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::MatrixFreeForwardReverseFlavor, ::Val{false}, ad_backend_fwd::TADFwd, ad_backend_rev::TADRev,
        f::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TADFwd<:ADTypes.AbstractADType, TADRev<:ADTypes.AbstractADType}
    can_jvp(ad_backend_fwd) || throw(ArgumentError("AD backend $(ad_backend_fwd) does not support JVPs (pushforwards), which are required for the forward direction of MatrixFreeForwardReverseFlavor"))
    can_vjp(ad_backend_rev) || throw(ArgumentError("AD backend $(ad_backend_rev) does not support VJPs (pullbacks), which are required for the reverse direction of MatrixFreeForwardReverseFlavor"))
    return ADImplicitComp(flavor, Val(false), ad_backend_fwd, ad_backend_rev, f, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_func=guess_nonlinear)
end
"""
    create_implicit_component(flavor::DerivativeFlavor, in_place::Bool, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing)

Create an [`ADImplicitComp`](@ref) from a user-defined function and state/output and input `ComponentVector`s.

`ad_backend` is validated against the flavor: [`DenseFlavor`](@ref) requires a dense (non-`AutoSparse`) backend; [`SparseFlavor`](@ref) requires an `AutoSparse` backend; [`MatrixFreeForwardFlavor`](@ref) requires a backend that supports JVPs (pushforwards); and [`MatrixFreeReverseFlavor`](@ref) requires a backend that supports VJPs (pullbacks).

# Positional Arguments
* `flavor`: [`DenseFlavor`](@ref), [`SparseFlavor`](@ref), [`MatrixFreeForwardFlavor`](@ref), or [`MatrixFreeReverseFlavor`](@ref)
* `in_place`: if `true`, the user-defined function writes its residuals in place; if `false`, it returns the residuals.
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentiation "backend" library
* `f`: the user-defined residual function. If `in_place` is `true`, a function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`. If `in_place` is `false`, a function of the form `R_ca = f(Y_ca, X_ca, params)` which returns the residuals.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed as the last argument to `f`. Could be anything, or `nothing`, but the derivatives of the residuals with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::DerivativeFlavor, in_place::Bool, ad_backend::TAD,
        f::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TAD<:ADTypes.AbstractADType}
    return create_implicit_component(flavor, Val(in_place), ad_backend, f, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_nonlinear=guess_nonlinear)
end

"""
    create_implicit_component(::MatrixFreeForwardReverseFlavor, in_place::Bool, ad_backend_fwd, ad_backend_rev, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing)

Create a [`MatrixFreeForwardReverseFlavor`](@ref) [`ADImplicitComp`](@ref) from a
user-defined function and state/output and input `ComponentVector`s. JVPs
(`mode="fwd"`) use `ad_backend_fwd`, which must support pushforwards; VJPs
(`mode="rev"`) use `ad_backend_rev`, which must support pullbacks.

# Positional Arguments
* `flavor`: [`MatrixFreeForwardReverseFlavor`](@ref)
* `in_place`: if `true`, the user-defined function writes its residuals in place; if `false`, it returns the residuals.
* `ad_backend_fwd`: `<:ADTypes.AbstractADType` backend used for forward-mode (JVP) derivatives
* `ad_backend_rev`: `<:ADTypes.AbstractADType` backend used for reverse-mode (VJP) derivatives
* `f`: the user-defined residual function. If `in_place` is `true`, a function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`. If `in_place` is `false`, a function of the form `R_ca = f(Y_ca, X_ca, params)` which returns the residuals.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed as the last argument to `f`. Could be anything, or `nothing`, but the derivatives of the residuals with respect to `params` will not be calculated.
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s).
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags.
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection.
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol.
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup.
* `guess_nonlinear`: optional function used to override OpenMDAO's `guess_nonlinear` (via [`guess_nonlinear!`](@ref)), providing initial guesses for the states before a nonlinear solve. For in-place comps, a function of the form `f_guess!(Y_ca, R_ca, X_ca, params)`, which receives the current states (which it overwrites with its state guesses), the current residuals, and the inputs. For out-of-place comps, a function of the form `Y_ca_new = f_guess(Y_ca_old, R_ca, X_ca, params)`, which returns the updated states as a single `ComponentVector`. OpenMDAO guarantees the residuals passed to a `guess_nonlinear` callback are fresh (computed from the current inputs and outputs), but also treats them as read-only there, so `R_ca` is provided for context only. The guess function is called (via `guess_nonlinear!`) when `guess_nonlinear` is not `nothing`, which also makes `has_guess_nonlinear` return `true` for the component.
"""
function create_implicit_component(flavor::MatrixFreeForwardReverseFlavor, in_place::Bool, ad_backend_fwd::TADFwd, ad_backend_rev::TADRev,
        f::Function, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_nonlinear=nothing) where {TADFwd<:ADTypes.AbstractADType, TADRev<:ADTypes.AbstractADType}
    return create_implicit_component(flavor, Val(in_place), ad_backend_fwd, ad_backend_rev, f, Y_ca, X_ca;
        params=params, units_dict=units_dict, tags_dict=tags_dict, shape_by_conn_dict=shape_by_conn_dict, copy_shape_dict=copy_shape_dict, force_skip_prep=force_skip_prep, guess_nonlinear=guess_nonlinear)
end
