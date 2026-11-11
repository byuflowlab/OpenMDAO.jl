# Dense-flavor implicit AD component: constructors and methods.
#
# Note: the prep builders from `dense_ad.jl` (`_get_dense_prep_stuff`) are
# reused for implicit components: the implicit `compute_adable(R, YX)`
# closures already have DifferentiationInterface's in-place (`f!(y, x)`) or
# out-of-place (`y = f(x)`) forms, so we pass `R_ca` as the "output" argument
# (`Y_ca`) and the combined `YX_ca` as the "input" argument (`X_ca`).

"""
    ADImplicitComp(::DenseFlavor, ::Val{true}, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an in-place [`DenseFlavor`](@ref) [`ADImplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f!`: function of the form `f!(R_ca, Y_ca, X_ca, params)` which writes residuals to `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the fourth argument to `f!`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup
"""
function ADImplicitComp(::DenseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    # Check for name collisions between output and input keys.
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    # Build the concatenated YX vector: outputs first, then inputs.
    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)

    # Precompute integer index ranges for Y and X within YX so the closure uses only
    # plain range indexing — compatible with all AD backends including ReverseDiff.
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    # Derive the residual vector from Y_ca (same structure).
    R_ca = similar(Y_ca)

    # Get the prep-related stuff. If any variables get their shapes from
    # connections (or prep is explicitly skipped), the shapes aren't known yet,
    # so defer everything to `setup_partials` (called by OpenMDAO during problem setup).
    compute_adable = _make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, R_ca, YX_ca)
    else
        deriv_prep = DenseDerivPrep(nothing, nothing)
        YX_ca_cs = R_ca_cs = nothing
    end

    return ADImplicitComp{DenseFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::DenseFlavor, ::Val{false}, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_skip_prep=false)

Create an out-of-place [`DenseFlavor`](@ref) [`ADImplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f`: function of the form `R_ca = f(Y_ca, X_ca, params)` which returns residuals `R_ca` using states/outputs `Y_ca`, inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of states/outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f`. Could be anything, or `nothing`, but the derivatives of `R_ca` with respect to `params` will not be calculated
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup
"""
function ADImplicitComp(::DenseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AbstractADType}
    # Check for name collisions between output and input keys.
    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    # Build the concatenated YX vector: outputs first, then inputs.
    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)

    # Precompute integer index ranges for Y and X within YX so the closure uses only
    # plain range indexing — compatible with all AD backends including ReverseDiff.
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = _make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, YX_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, YX_ca)
    else
        deriv_prep = DenseDerivPrep(nothing, nothing)
        YX_ca_cs = nothing
    end

    R_ca = nothing
    R_ca_cs = nothing

    return ADImplicitComp{DenseFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=guess_func)
end

function _update_prep(comp::ADImplicitComp{DenseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = get_output_ca(comp)
        X_ca_old = get_input_ca(comp)

        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        R_ca = similar(Y_ca)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = _make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_dense_prep_stuff(get_backend(comp), compute_adable, R_ca, YX_ca)

        comp = ADImplicitComp{DenseFlavor, true}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function _update_prep(comp::ADImplicitComp{DenseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
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
        deriv_prep, YX_ca_cs = _get_dense_prep_stuff(get_backend(comp), compute_adable, YX_ca)

        R_ca = nothing
        R_ca_cs = nothing

        comp = ADImplicitComp{DenseFlavor, false}(get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, InputCopyPrep(Y_ca), InputCopyPrep(X_ca), R_ca === nothing ? InputCopyPrep(Y_ca) : InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function linearize!(comp::ADImplicitComp{DenseFlavor, true}, inputs, outputs, partials)
    YX_ca = get_combined_ca(comp)
    # Copy outputs (states) and inputs into the combined YX vector
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Single jacobian! call for the full [dR/dY | dR/dX] block.
    f! = get_callback(comp)
    R_ca = get_residual_ca(comp)
    J_ca = get_jacobian_ca(comp)
    prep = get_prep(comp)
    ad_backend = get_backend(comp)
    if J_ca === nothing
        throw(_prep_not_created_error(comp, "linearize!"))
    elseif prep === nothing
        DifferentiationInterface.jacobian!(f!, R_ca, J_ca, ad_backend, YX_ca)
    else
        DifferentiationInterface.jacobian!(f!, R_ca, J_ca, prep, ad_backend, YX_ca)
    end

    _copy_dense_partials!(partials, J_ca, comp.deriv_prep.partial_copy_prep)
    return nothing
end

function linearize!(comp::ADImplicitComp{DenseFlavor, false}, inputs, outputs, partials)
    YX_ca = get_combined_ca(comp)
    # Copy outputs (states) and inputs into the combined YX vector
    _copy_component_inputs!(view(getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    _copy_component_inputs!(view(getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    # Single jacobian! call for the full [dR/dY | dR/dX] block.
    f = get_callback(comp)
    J_ca = get_jacobian_ca(comp)
    prep = get_prep(comp)
    ad_backend = get_backend(comp)
    if J_ca === nothing
        throw(_prep_not_created_error(comp, "linearize!"))
    elseif prep === nothing
        DifferentiationInterface.jacobian!(f, J_ca, ad_backend, YX_ca)
    else
        DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, YX_ca)
    end

    _copy_dense_partials!(partials, J_ca, comp.deriv_prep.partial_copy_prep)
    return nothing
end
