# Matrix-free-flavor implicit AD component: constructors and methods.
#
# Note: the prep builders from `matrix_free_ad.jl`
# (`_get_matrix_free_forward_prep_in_place`, etc.) are reused for implicit
# components: the implicit `compute_adable(R, YX)` closures already have
# DifferentiationInterface's in-place (`f!(y, x)`) or out-of-place
# (`y = f(x)`) forms, so we pass `R_ca` as the "output" argument (`Y_ca`) and
# the combined `YX_ca` as the "input" argument (`X_ca`). In the resulting
# `MatrixFreeDerivPrep`, `dX_ca` holds the combined dYX tangent/cotangent
# buffer and `dY_ca` holds the residual tangent/cotangent buffer dR.

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    R_ca = similar(Y_ca)

    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_prep_in_place(
            ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    return ADImplicitComp{MatrixFreeForwardFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pushforward!`.
"""
function ADImplicitComp(::MatrixFreeForwardFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_prep_out_of_place(
            ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeForwardFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    R_ca = similar(Y_ca)

    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_reverse_prep_in_place(
            ad_backend, compute_adable, R_ca, YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    return ADImplicitComp{MatrixFreeReverseFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.pullback!`.
"""
function ADImplicitComp(::MatrixFreeReverseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs = _get_matrix_free_reverse_prep_out_of_place(
            ad_backend, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeDerivPrep(dYX_ca, dR_ca, nothing)
    end

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeReverseFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::MatrixFreeForwardReverseFlavor, ::Val{true}, ad_backend_fwd, ad_backend_rev, f!, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an in-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
JVPs (`mode="fwd"`) are computed via `DifferentiationInterface.pushforward!` using
`ad_backend_fwd`; VJPs (`mode="rev"`) are computed via `DifferentiationInterface.pullback!`
using `ad_backend_rev`.
"""
function ADImplicitComp(::MatrixFreeForwardReverseFlavor, ::Val{true}, ad_backend_fwd, ad_backend_rev, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing)
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    R_ca = similar(Y_ca)

    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_reverse_prep_in_place(
            ad_backend_fwd, ad_backend_rev, compute_adable, R_ca, YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeForwardReverseDerivPrep(dYX_ca, dR_ca, nothing, nothing)
    end

    return ADImplicitComp{MatrixFreeForwardReverseFlavor, true}(ad_backend_fwd, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); ad_backend_rev=ad_backend_rev, guess_func=guess_func)
end

"""
    ADImplicitComp(::MatrixFreeForwardReverseFlavor, ::Val{false}, ad_backend_fwd, ad_backend_rev, f, Y_ca, X_ca; params=nothing, force_skip_prep=false)

Create an out-of-place [`MatrixFreeForwardReverseFlavor`](@ref) [`ADImplicitComp`](@ref).
JVPs (`mode="fwd"`) are computed via `DifferentiationInterface.pushforward!` using
`ad_backend_fwd`; VJPs (`mode="rev"`) are computed via `DifferentiationInterface.pullback!`
using `ad_backend_rev`.
"""
function ADImplicitComp(::MatrixFreeForwardReverseFlavor, ::Val{false}, ad_backend_fwd, ad_backend_rev, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing)
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_reverse_prep_out_of_place(
            ad_backend_fwd, ad_backend_rev, compute_adable, compute_adable(YX_ca), YX_ca, force_skip_prep)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup). The prep-less DifferentiationInterface methods are
        # used at runtime until then.
        dYX_ca = ComponentVector{eltype(YX_ca)}()
        dR_ca = ComponentVector{eltype(Y_ca)}()
        YX_ca_cs = ComponentVector{ComplexF64}()
        R_ca_cs = ComponentVector{ComplexF64}()
        deriv_prep = MatrixFreeForwardReverseDerivPrep(dYX_ca, dR_ca, nothing, nothing)
    end

    R_ca = R_ca_cs = nothing

    return ADImplicitComp{MatrixFreeForwardReverseFlavor, false}(ad_backend_fwd, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); ad_backend_rev=ad_backend_rev, guess_func=guess_func)
end

function _update_prep(comp::ADImplicitComp{MatrixFreeForwardFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = similar(Y_ca)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_prep_in_place(ad_backend, compute_adable, R_ca, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardFlavor, true}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{<:MatrixFreeFlavor, true}, inputs, outputs, dinputs, doutputs, dresids, prep, backend)
    YX_ca = get_combined_ca(comp)
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Assemble the YX tangent from doutputs (Y part) and dinputs (X part).
    dYX_ca = get_dcombined_ca(comp)
    _copy_component_deriv_inputs!(view(getdata(dYX_ca), comp.Y_range), doutputs, comp.state_copy_prep)
    _copy_component_deriv_inputs!(view(getdata(dYX_ca), comp.X_range), dinputs, comp.input_copy_prep)

    R_ca = get_residual_ca(comp)
    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    if prep === nothing
        isempty(dYX_ca) && throw(_prep_not_created_error(comp, "apply_linear!"))
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pushforward!(compute_adable, R_ca, (dR_ca,), backend, YX_ca, (dYX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, R_ca, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))
    end

    _copy_component_outputs_accum!(getdata(dR_ca), dresids, comp.residual_copy_prep)

    return nothing
end

function _update_prep(comp::ADImplicitComp{MatrixFreeForwardFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = nothing
        R_ca_cs = nothing
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_prep_out_of_place(ad_backend, compute_adable, similar(Y_ca), YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardFlavor, false}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, nothing, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), InputCopyPrep(Y_ca); guess_func=comp.guess_func)
    end

    return comp
end

function _compute_implicit_pushforward!(comp::ADImplicitComp{<:MatrixFreeFlavor, false}, inputs, outputs, dinputs, doutputs, dresids, prep, backend)
    YX_ca = get_combined_ca(comp)
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    dYX_ca = get_dcombined_ca(comp)
    _copy_component_deriv_inputs!(view(getdata(dYX_ca), comp.Y_range), doutputs, comp.state_copy_prep)
    _copy_component_deriv_inputs!(view(getdata(dYX_ca), comp.X_range), dinputs, comp.input_copy_prep)

    dR_ca = get_dresidual_ca(comp)
    compute_adable = get_callback(comp)
    if prep === nothing
        isempty(dYX_ca) && throw(_prep_not_created_error(comp, "apply_linear!"))
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pushforward!(compute_adable, (dR_ca,), backend, YX_ca, (dYX_ca,))
    else
    DifferentiationInterface.pushforward!(compute_adable, (dR_ca,), prep, backend, YX_ca, (dYX_ca,))
    end

    _copy_component_outputs_accum!(getdata(dR_ca), dresids, comp.residual_copy_prep)

    return nothing
end

function _update_prep(comp::ADImplicitComp{MatrixFreeReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = similar(Y_ca)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_reverse_prep_in_place(ad_backend, compute_adable, R_ca, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeReverseFlavor, true}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function _compute_implicit_pullback!(comp::ADImplicitComp{<:MatrixFreeFlavor, true}, inputs, outputs, dinputs, doutputs, dresids, prep, backend)
    YX_ca = get_combined_ca(comp)
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Load the residual cotangent seed from dresids.
    dR_ca = get_dresidual_ca(comp)
    _copy_component_inputs!(getdata(dR_ca), dresids, comp.residual_copy_prep)

    R_ca = get_residual_ca(comp)
    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    if prep === nothing
        isempty(dYX_ca) && throw(_prep_not_created_error(comp, "apply_linear!"))
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pullback!(compute_adable, R_ca, (dYX_ca,), backend, YX_ca, (dR_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, R_ca, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))
    end

    # Scatter dYX cotangents back: Y part → doutputs, X part → dinputs.
    _copy_component_outputs_accum!(view(getdata(dYX_ca), comp.Y_range), doutputs, comp.state_copy_prep)
    _copy_component_outputs_accum!(view(getdata(dYX_ca), comp.X_range), dinputs, comp.input_copy_prep)

    return nothing
end

function _update_prep(comp::ADImplicitComp{MatrixFreeReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        ad_backend = get_backend(comp)
        R_ca = nothing
        R_ca_cs = nothing
        deriv_prep, YX_ca_cs = _get_matrix_free_reverse_prep_out_of_place(ad_backend, compute_adable, similar(Y_ca), YX_ca, false)

        comp = ADImplicitComp{MatrixFreeReverseFlavor, false}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, nothing, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), InputCopyPrep(Y_ca); guess_func=comp.guess_func)
    end

    return comp
end

function _update_prep(comp::ADImplicitComp{MatrixFreeForwardReverseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        backend_fwd = get_backend_fwd(comp)
        backend_rev = get_backend_rev(comp)
        R_ca = similar(Y_ca)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_matrix_free_forward_reverse_prep_in_place(backend_fwd, backend_rev, compute_adable, R_ca, YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardReverseFlavor, true}(backend_fwd, compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); ad_backend_rev=backend_rev, guess_func=comp.guess_func)
    end

    return comp
end

function _update_prep(comp::ADImplicitComp{MatrixFreeForwardReverseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        backend_fwd = get_backend_fwd(comp)
        backend_rev = get_backend_rev(comp)
        R_ca = nothing
        R_ca_cs = nothing
        deriv_prep, YX_ca_cs = _get_matrix_free_forward_reverse_prep_out_of_place(backend_fwd, backend_rev, compute_adable, similar(Y_ca), YX_ca, false)

        comp = ADImplicitComp{MatrixFreeForwardReverseFlavor, false}(backend_fwd, compute_adable, comp.func, comp.params, YX_ca, nothing, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), InputCopyPrep(Y_ca); ad_backend_rev=backend_rev, guess_func=comp.guess_func)
    end

    return comp
end

function _compute_implicit_pullback!(comp::ADImplicitComp{<:MatrixFreeFlavor, false}, inputs, outputs, dinputs, doutputs, dresids, prep, backend)
    YX_ca = get_combined_ca(comp)
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    dR_ca = get_dresidual_ca(comp)
    _copy_component_inputs!(getdata(dR_ca), dresids, comp.residual_copy_prep)

    dYX_ca = get_dcombined_ca(comp)
    compute_adable = get_callback(comp)
    if prep === nothing
        isempty(dYX_ca) && throw(_prep_not_created_error(comp, "apply_linear!"))
        # No prep available (e.g. `force_skip_prep=true`): fall back to the
        # prep-less DifferentiationInterface methods, which prepare internally
        # on each call.
    DifferentiationInterface.pullback!(compute_adable, (dYX_ca,), backend, YX_ca, (dR_ca,))
    else
    DifferentiationInterface.pullback!(compute_adable, (dYX_ca,), prep, backend, YX_ca, (dR_ca,))
    end

    _copy_component_outputs_accum!(view(getdata(dYX_ca), comp.Y_range), doutputs, comp.state_copy_prep)
    _copy_component_outputs_accum!(view(getdata(dYX_ca), comp.X_range), dinputs, comp.input_copy_prep)

    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeForwardFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "fwd"
        _compute_implicit_pushforward!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, get_prep(comp), get_backend(comp))
    else
        throw(ArgumentError("MatrixFreeForwardFlavor only supports mode=\"fwd\""))
    end
    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeReverseFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "rev"
        _compute_implicit_pullback!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, get_prep(comp), get_backend(comp))
    else
        throw(ArgumentError("MatrixFreeReverseFlavor only supports mode=\"rev\""))
    end
    return nothing
end

function apply_linear!(comp::ADImplicitComp{MatrixFreeForwardReverseFlavor}, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    if mode == "fwd"
        _compute_implicit_pushforward!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, comp.deriv_prep.prep_fwd, get_backend_fwd(comp))
    elseif mode == "rev"
        _compute_implicit_pullback!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, comp.deriv_prep.prep_rev, get_backend_rev(comp))
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end
    return nothing
end
