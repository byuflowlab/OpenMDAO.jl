# Dense-flavor explicit AD component: constructors and methods.
#
# The `ADExplicitComp{DenseFlavor, ...}` *type* and the shared accessors are
# declared in `abstract_ad.jl`. This file provides the flavor-specific
# constructors and the `compute_partials!`/`setup_partials`/`_update_prep`/
# `get_partials_data` methods.

using ADTypes: ADTypes
using ComponentArrays: ComponentVector, ComponentMatrix, getaxes
using DifferentiationInterface: DifferentiationInterface

# Note: these prep builders are also reused by the implicit AD components in
# `dense_ad_implicit.jl`: the implicit `compute_adable(R, YX)` closures already
# have DifferentiationInterface's in-place (`f!(y, x)`) or out-of-place
# (`y = f(x)`) forms, so implicit components pass `R_ca` as the "output"
# argument (`Y_ca`) and the combined `YX_ca` as the "input" argument (`X_ca`).
function _get_dense_prep_stuff(ad_backend, f!, Y_ca, X_ca, force_skip_prep::Bool=false)
    # Need to "prepare" the backend if not skipping prep.
    prep = force_skip_prep ? nothing : DifferentiationInterface.prepare_jacobian(f!, Y_ca, ad_backend, X_ca)

    # Get the Jacobian matrix.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    TCS = Complex{TF}
    X_ca_cs = similar(X_ca, TCS)
    Y_ca_cs = similar(Y_ca, TCS)

    return DenseDerivPrep(J_ca, prep), X_ca_cs, Y_ca_cs
end

function _get_dense_prep_stuff(ad_backend, f, X_ca, force_skip_prep::Bool=false)
    # Need to "prepare" the backend if not skipping prep.
    prep = force_skip_prep ? nothing : DifferentiationInterface.prepare_jacobian(f, ad_backend, X_ca)

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = f(X_ca)

    # Now I think I can get the sparse Jacobian from that.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return DenseDerivPrep(J_ca, prep), X_ca_cs
end

"""
    ADExplicitComp(::DenseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an in-place [`DenseFlavor`](@ref) [`ADExplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup
"""
function ADExplicitComp(::DenseFlavor, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}

    # Get the prep-related stuff.
    compute_adable = _make_compute_adable(Val(true), f!, params)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        deriv_prep = DenseDerivPrep(nothing, nothing)
        X_ca_cs = Y_ca_cs = nothing
    end

    return ADExplicitComp{DenseFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
end

"""
    ADExplicitComp(::DenseFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an out-of-place [`DenseFlavor`](@ref) [`ADExplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `force_skip_prep`: if true, defer creating internal arrays and other structs until OpenMDAO calls `setup_partials` during problem setup
"""
function ADExplicitComp(::DenseFlavor, ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AbstractADType}

    compute_adable = _make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, X_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        deriv_prep = DenseDerivPrep(nothing, nothing)
        X_ca_cs = nothing
    end

    Y_ca_cs = nothing
    return ADExplicitComp{DenseFlavor, false}(ad_backend, f, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
end

# `get_jacobian_ca` for `DenseFlavor` is defined in `abstract_ad.jl`.

function _update_prep(self::ADExplicitComp{DenseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        # For an out-of-place component, this will call the callback function on self.X_ca, which I think should be fine.
        # Ah, but this is an in-place component anyway.
        Y_ca_old = get_output_ca(self)

        # Create a new versions of `X_ca_old` that have the correct sizes and default values.
        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        compute_adable = _make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = ADExplicitComp{DenseFlavor, true}(ad_backend, self.func, self.params, compute_adable, X_ca, Y_ca, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, InputCopyPrep(X_ca), InputCopyPrep(Y_ca))
    end

    return self
end

function _update_prep(self::ADExplicitComp{DenseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if length(input_sizes) > 0
        X_ca_old = get_input_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        compute_adable = _make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict

        self = ADExplicitComp{DenseFlavor, false}(ad_backend, self.func, self.params, compute_adable, X_ca, nothing, deriv_prep,
            units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, nothing, InputCopyPrep(X_ca), InputCopyPrep(compute_adable(X_ca)))
    end

    return self
end

function get_partials_data(self::ADExplicitComp{DenseFlavor})
    return [OpenMDAOCore.PartialsData("*", "*")]
end

function setup_partials(self::ADExplicitComp{DenseFlavor}, input_sizes, output_sizes)

    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = _update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, get_partials_data(self_new)
end

function OpenMDAOCore.compute_partials!(self::ADExplicitComp{DenseFlavor, true}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    # Get the Jacobian.
    f! = get_callback(self)
    Y_ca = get_output_ca(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    prep === nothing && throw(_prep_not_created_error(self, "compute_partials!"))
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    _copy_dense_partials!(partials, J_ca, self.deriv_prep.partial_copy_prep)

    return nothing
end

function OpenMDAOCore.compute_partials!(self::ADExplicitComp{DenseFlavor, false}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    _copy_component_inputs!(getdata(X_ca), inputs, self.input_copy_prep)

    # Get the Jacobian.
    f = get_callback(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    prep === nothing && throw(_prep_not_created_error(self, "compute_partials!"))
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    _copy_dense_partials!(partials, J_ca, self.deriv_prep.partial_copy_prep)

    return nothing
end
