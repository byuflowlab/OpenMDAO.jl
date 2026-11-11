# Matrix-free explicit AD component: constructors and methods, split into
# `MatrixFreeForwardFlavor` (pushforward / JVP) and `MatrixFreeReverseFlavor`
# (pullback / VJP).
#
# The `ADExplicitComp{<:MatrixFreeFlavor, ...}` *type* and the shared accessors
# (`get_dinput_ca`, `get_doutput_ca`, `has_*`) are declared in `abstract_ad.jl`.
# This file provides the flavor-specific constructors, `_update_prep`,
# `setup_partials`, and `compute_jacvec_product!` (+ internals).

using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using DifferentiationInterface: DifferentiationInterface

# ---------------------------------------------------------------------------
# Flavor auto-detection (replaces the old `force_mode=""` behavior).
# ---------------------------------------------------------------------------

# Resolve the old `force_mode` argument to a concrete matrix-free flavor.
# `force_mode="fwd"` -> `MatrixFreeForwardFlavor`; `"rev"` -> `MatrixFreeReverseFlavor`;
# `""` -> pick based on the backend's pushforward performance (preferring forward),
# matching the previous runtime auto-detection.
function _resolve_matrix_free_flavor(ad_backend, force_mode::AbstractString="")
    if force_mode == "fwd"
        return MatrixFreeForwardFlavor()
    elseif force_mode == "rev"
        return MatrixFreeReverseFlavor()
    elseif force_mode == ""
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            return MatrixFreeForwardFlavor()
        else
            return MatrixFreeReverseFlavor()
        end
    else
        throw(ArgumentError("force_mode argument should be one of `\"\"`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end
end

# ---------------------------------------------------------------------------
# Prep builders. `force_skip_prep` replaces the old `disable_prep` flag.
# ---------------------------------------------------------------------------

# The tangent/cotangent buffers for the matrix-free flavors are plain arrays of
# the same shape and element type as the primal vectors, for both derivative
# directions, so a single routine creates them for all flavor/in-place
# combinations.
_get_matrix_free_buffers(X_ca, Y_ca) = (similar(X_ca), similar(Y_ca))

function _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    Y_ca_cs = similar(Y_ca, Complex{eltype(Y_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs, Y_ca_cs
end

function _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    Y_ca_cs = similar(Y_ca, Complex{eltype(Y_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs, Y_ca_cs
end

function _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs
end

function _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep = nothing
    else
        prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    return MatrixFreeDerivPrep(dX_ca, dY_ca, prep), X_ca_cs
end

# Prep builders for `MatrixFreeForwardReverseFlavor`: separate forward- and
# reverse-mode backends share the same plain-Float64 tangent/cotangent buffers.
function _get_matrix_free_forward_reverse_prep_in_place(backend_fwd, backend_rev, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep_fwd = nothing
        prep_rev = nothing
    else
        prep_fwd = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, backend_fwd, X_ca, (dX_ca,))
        prep_rev = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, backend_rev, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    Y_ca_cs = similar(Y_ca, Complex{eltype(Y_ca)})
    return MatrixFreeForwardReverseDerivPrep(dX_ca, dY_ca, prep_fwd, prep_rev), X_ca_cs, Y_ca_cs
end

function _get_matrix_free_forward_reverse_prep_out_of_place(backend_fwd, backend_rev, compute_adable, Y_ca, X_ca, force_skip_prep::Bool)
    dX_ca, dY_ca = _get_matrix_free_buffers(X_ca, Y_ca)
    if force_skip_prep
        prep_fwd = nothing
        prep_rev = nothing
    else
        prep_fwd = DifferentiationInterface.prepare_pushforward(compute_adable, backend_fwd, X_ca, (dX_ca,))
        prep_rev = DifferentiationInterface.prepare_pullback(compute_adable, backend_rev, X_ca, (dY_ca,))
    end
    X_ca_cs = similar(X_ca, Complex{eltype(X_ca)})
    return MatrixFreeForwardReverseDerivPrep(dX_ca, dY_ca, prep_fwd, prep_rev), X_ca_cs
end

# ---------------------------------------------------------------------------
# Constructors.
# ---------------------------------------------------------------------------

"""
    ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`MatrixFreeForwardFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!` (JVPs).
"""
function ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(true), f!, params)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    return ADExplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
end

"""
    ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`MatrixFreeReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!` (VJPs).
"""
function ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(true), f!, params)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    return ADExplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
end

"""
    ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!` (JVPs).
"""
function ADExplicitComp(::MatrixFreeForwardFlavor, ad_backend, f, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    Y_ca_cs = nothing
    return ADExplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, f, params, compute_adable, X_ca, nothing, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(compute_adable(X_ca)))
end

"""
    ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`MatrixFreeReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!` (VJPs).
"""
function ADExplicitComp(::MatrixFreeReverseFlavor, ad_backend, f, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The prep is set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real prep.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dX_ca, dY_ca, nothing)
    end

    Y_ca_cs = nothing
    return ADExplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, f, params, compute_adable, X_ca, nothing, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(compute_adable(X_ca)))
end

"""
    ADExplicitComp(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
JVPs (`mode="fwd"`) are computed via `DifferentiationInterface.pushforward!` using
`ad_backend_fwd`; VJPs (`mode="rev"`) are computed via `DifferentiationInterface.pullback!`
using `ad_backend_rev`.
"""
function ADExplicitComp(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(true), f!, params)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_reverse_prep_in_place(ad_backend_fwd, ad_backend_rev, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The preps are set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real preps.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeForwardReverseDerivPrep(dX_ca, dY_ca, nothing, nothing)
    end

    return ADExplicitComp{MatrixFreeForwardReverseFlavor, true}(ad_backend_fwd, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca);
        ad_backend_rev=ad_backend_rev)
end

"""
    ADExplicitComp(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADExplicitComp`](@ref).
JVPs (`mode="fwd"`) are computed via `DifferentiationInterface.pushforward!` using
`ad_backend_fwd`; VJPs (`mode="rev"`) are computed via `DifferentiationInterface.pullback!`
using `ad_backend_rev`.
"""
function ADExplicitComp(::MatrixFreeForwardReverseFlavor, ad_backend_fwd, ad_backend_rev, f, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(),
        shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_reverse_prep_out_of_place(ad_backend_fwd, ad_backend_rev, compute_adable, Y_ca, X_ca, force_skip_prep)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        # The preps are set to `nothing`; the prep-less DifferentiationInterface
        # methods are used at runtime until OpenMDAO's `setup_partials` call
        # creates the real preps.
        dX_ca = ComponentVector{eltype(X_ca)}()
        dY_ca = ComponentVector{eltype(Y_ca)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeForwardReverseDerivPrep(dX_ca, dY_ca, nothing, nothing)
    end

    Y_ca_cs = nothing
    return ADExplicitComp{MatrixFreeForwardReverseFlavor, false}(ad_backend_fwd, f, params, compute_adable, X_ca, nothing, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(compute_adable(X_ca));
        ad_backend_rev=ad_backend_rev)
end

# ---------------------------------------------------------------------------
# _update_prep / setup_partials / get_partials_data
# ---------------------------------------------------------------------------

function _update_prep(self::ADExplicitComp{MatrixFreeForwardFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f! = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, f!, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, self.func, self.params, f!, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f! = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, f!, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, self.func, self.params, f!, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeForwardFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, f, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, self.func, self.params, f, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        ad_backend = get_backend(self)
        f = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, f, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, self.func, self.params, f, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeForwardReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        backend_fwd = get_backend_fwd(self)
        backend_rev = get_backend_rev(self)
        f! = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_matrix_free_forward_reverse_prep_in_place(backend_fwd, backend_rev, f!, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardReverseFlavor, true}(backend_fwd, self.func, self.params, f!, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca);
            ad_backend_rev=backend_rev)
    end
    return self
end

function _update_prep(self::ADExplicitComp{MatrixFreeForwardReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        backend_fwd = get_backend_fwd(self)
        backend_rev = get_backend_rev(self)
        f = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_matrix_free_forward_reverse_prep_out_of_place(backend_fwd, backend_rev, f, Y_ca, X_ca, false)

        self = ADExplicitComp{MatrixFreeForwardReverseFlavor, false}(backend_fwd, self.func, self.params, f, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing, InputCopyPrep(X_ca), InputCopyPrep(Y_ca);
            ad_backend_rev=backend_rev)
    end
    return self
end

# Matrix-free components don't declare any partials (the Jacobian is never assembled).
function get_partials_data(self::ADExplicitComp{<:MatrixFreeFlavor})
    return Vector{PartialsData}()
end

function setup_partials(self::ADExplicitComp{<:MatrixFreeFlavor}, input_sizes, output_sizes)
    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = _update_prep(self, input_sizes_ca, output_sizes_ca)

    return self_new, get_partials_data(self_new)
end

# ---------------------------------------------------------------------------
# compute_jacvec_product! + internals (pushforward / pullback).
# ---------------------------------------------------------------------------

function _compute_pushforward!(self::ADExplicitComp{<:MatrixFreeFlavor, true}, inputs, d_inputs, d_outputs, prep, backend)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    # d_inputs has the derivative of each input wrt some upstream input.
    dX_ca = get_dinput_ca(self)
    _copy_component_deriv_inputs!(getdata(dX_ca), d_inputs, self.input_copy_prep)

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && throw(_prep_not_created_error(self, "compute_jacvec_product!"))
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pushforward!(compute_adable, Y_ca, (dY_ca,), backend, X_ca, (dX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, Y_ca, (dY_ca,), prep, backend, X_ca, (dX_ca,))
    end

    # Now copy the output derivatives to `d_outputs`:
    _copy_component_outputs_accum!(getdata(dY_ca), d_outputs, self.output_copy_prep)

    return nothing
end

function _compute_pushforward!(self::ADExplicitComp{<:MatrixFreeFlavor, false}, inputs, d_inputs, d_outputs, prep, backend)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    dX_ca = get_dinput_ca(self)
    _copy_component_deriv_inputs!(getdata(dX_ca), d_inputs, self.input_copy_prep)

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && throw(_prep_not_created_error(self, "compute_jacvec_product!"))
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pushforward!(compute_adable, (dY_ca,), backend, X_ca, (dX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, (dY_ca,), prep, backend, X_ca, (dX_ca,))
    end

    # Now copy the output derivatives to `d_outputs`:
    _copy_component_outputs_accum!(getdata(dY_ca), d_outputs, self.output_copy_prep)

    return nothing
end

function _compute_pullback!(self::ADExplicitComp{<:MatrixFreeFlavor, true}, inputs, d_inputs, d_outputs, prep, backend)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    dY_ca = get_doutput_ca(self)
    _copy_component_deriv_inputs!(getdata(dY_ca), d_outputs, self.output_copy_prep)

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && throw(_prep_not_created_error(self, "compute_jacvec_product!"))
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pullback!(compute_adable, Y_ca, (dX_ca,), backend, X_ca, (dY_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, Y_ca, (dX_ca,), prep, backend, X_ca, (dY_ca,))
    end

    # Now copy the input derivatives to `d_inputs`:
    _copy_component_outputs_accum!(getdata(dX_ca), d_inputs, self.input_copy_prep)

    return nothing
end

function _compute_pullback!(self::ADExplicitComp{<:MatrixFreeFlavor, false}, inputs, d_inputs, d_outputs, prep, backend)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    dY_ca = get_doutput_ca(self)
    _copy_component_deriv_inputs!(getdata(dY_ca), d_outputs, self.output_copy_prep)

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Now actually do the Jacobian-vector product.
    if prep === nothing
        isempty(dX_ca) && throw(_prep_not_created_error(self, "compute_jacvec_product!"))
        # No prep available: fall back to the prep-less DifferentiationInterface
        # methods, which prepare internally on each call.
    DifferentiationInterface.pullback!(compute_adable, (dX_ca,), backend, X_ca, (dY_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, (dX_ca,), prep, backend, X_ca, (dY_ca,))
    end

    # Now copy the input derivatives to `d_inputs`:
    _copy_component_outputs_accum!(getdata(dX_ca), d_inputs, self.input_copy_prep)

    return nothing
end

"""
    compute_jacvec_product!(self::ADExplicitComp{<:MatrixFreeFlavor}, inputs, d_inputs, d_outputs, mode)

Compute the Jacobian-vector product (`mode="fwd"`, requires
[`MatrixFreeForwardFlavor`](@ref) or [`MatrixFreeForwardReverseFlavor`](@ref)) or
vector-Jacobian product (`mode="rev"`, requires [`MatrixFreeReverseFlavor`](@ref)
or [`MatrixFreeForwardReverseFlavor`](@ref)).
"""
function OpenMDAOCore.compute_jacvec_product!(self::ADExplicitComp{<:MatrixFreeFlavor}, inputs, d_inputs, d_outputs, mode)
    backend = get_backend(self)
    prep = get_prep(self)
    if mode == "fwd"
        if self isa ADExplicitComp{MatrixFreeForwardFlavor}
            _compute_pushforward!(self, inputs, d_inputs, d_outputs, get_prep(self), get_backend(self))
        else
            @warn "mode = \"fwd\" not supported for AD backend $(nameof(typeof(backend))), preparation $(nameof(typeof(prep))), derivatives for $(nameof(typeof(self))) will be incorrect"
        end
    elseif mode == "rev"
        if self isa ADExplicitComp{MatrixFreeReverseFlavor}
            _compute_pullback!(self, inputs, d_inputs, d_outputs, get_prep(self), get_backend(self))
        else
            @warn "mode = \"rev\" not supported for AD backend $(nameof(typeof(backend))), preparation $(nameof(typeof(prep))), derivatives for $(nameof(typeof(self))) will be incorrect"
        end
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end

    return nothing
end

"""
    compute_jacvec_product!(self::ADExplicitComp{MatrixFreeForwardReverseFlavor}, inputs, d_inputs, d_outputs, mode)

Compute the Jacobian-vector product (`mode="fwd"`, using the forward-mode
backend) or vector-Jacobian product (`mode="rev"`, using the reverse-mode
backend); both directions are supported.
"""
function OpenMDAOCore.compute_jacvec_product!(self::ADExplicitComp{MatrixFreeForwardReverseFlavor}, inputs, d_inputs, d_outputs, mode)
    if mode == "fwd"
        _compute_pushforward!(self, inputs, d_inputs, d_outputs, self.deriv_prep.prep_fwd, get_backend_fwd(self))
    elseif mode == "rev"
        _compute_pullback!(self, inputs, d_inputs, d_outputs, self.deriv_prep.prep_rev, get_backend_rev(self))
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end

    return nothing
end
