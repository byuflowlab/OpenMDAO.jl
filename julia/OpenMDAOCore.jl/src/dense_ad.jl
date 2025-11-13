"""
    DenseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS,TAMD} <: AbstractExplicitComp{InPlace}

An `<:AbstractADExplicitComp` for dense Jacobians.

# Fields
* `ad_backend::TAD`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `compute_adable::TCompute`: function of the form `compute_adable(Y, X)` or `Y = compute_adable(x)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::ComponentVector`: `ComponentVector` of inputs
* `Y_ca::ComponentVector`: `ComponentVector` of outputs
* `J_ca::ComponentMatrix`: Dense `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `shape_by_conn_dict::Dict{Symbol,Bool}`: mapping of variable names to `Bool` indicating if the variable shape should be determined dynamically by a connection.
* `copy_shape_dict::Dict{Symbol,Symbol}`: mapping of variable names to variable names indicating if a variable shape should be copied from another variable.
* `aviary_input_names::Dict{Symbol,String}`: mapping of input variable names to Aviary names.
* `aviary_output_names::Dict{Symbol,String}`: mapping of output variable names to Aviary names.
* `aviary_meta_data::Dict{String,Any}`: mapping of Aviary variable names to aviary metadata. Currently only the `"units"` and `"default_value"` fields are used.
* `prep::DifferentiationInterface.JacobianPrep`: `DifferentiationInterface.jl` "preparation" object
* `X_ca::ComponentVector`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::ComponentVector`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct DenseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS,TAMD} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca::TJ
    prep::TPrep
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS
    aviary_input_names::Dict{Symbol,String}
    aviary_output_names::Dict{Symbol,String}
    aviary_meta_data::TAMD

    function DenseADExplicitComp{InPlace}(ad_backend, compute_adable, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data) where {InPlace}
        return new{InPlace, typeof(ad_backend), typeof(compute_adable), typeof(X_ca), typeof(Y_ca),
                   typeof(J_ca), typeof(prep),
                   typeof(X_ca_cs), typeof(Y_ca_cs),
                   typeof(aviary_meta_data)}(ad_backend,
                                             compute_adable, X_ca,
                                             Y_ca, J_ca,
                                             prep,
                                             units_dict,
                                             tags_dict,
                                             shape_by_conn_dict,
                                             copy_shape_dict,
                                             X_ca_cs, Y_ca_cs,
                                             aviary_input_names,
                                             aviary_output_names,
                                             aviary_meta_data)
    end
end

function DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    Y_ca = nothing
    Y_ca_cs = nothing
    return DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end

"""
    DenseADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, aviary_input_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_output_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_meta_data=Dict{String,Any}())

Create a `DenseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

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
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep!`
* `aviary_input_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of input variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary input variables.
* `aviary_output_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of output variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary output variables.
* `aviary_meta_data::Dict{String,Any}`: mapping of Aviary variable names to aviary metadata. Currently only the `"units"` and `"default_value"` fields are used.
"""
function DenseADExplicitComp(ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, aviary_input_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_output_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_meta_data=Dict{String,Nothing}()) where {TAD<:ADTypes.AbstractADType}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # Process the Aviary metadata.
    X_ca_full, units_dict_tmp, aviary_input_names = _process_aviary_metadata(X_ca, units_dict, aviary_input_vars, aviary_meta_data)
    Y_ca_full, units_dict_full, aviary_output_names = _process_aviary_metadata(Y_ca, units_dict_tmp, aviary_output_vars, aviary_meta_data)

    _check_aviary_names(aviary_input_names, aviary_output_names)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        prep, J_ca, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, Y_ca_full, X_ca_full)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        prep = J_ca = X_ca_cs = Y_ca_cs = nothing
    end

    return DenseADExplicitComp{true}(ad_backend, compute_adable, X_ca_full, Y_ca_full, J_ca, prep, units_dict_full, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end

"""
    DenseADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, aviary_input_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_output_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_meta_data=Dict{String,Any}())

Create a `DenseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

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
* `force_skip_prep`: if true, defer creating internal arrays and other structs until the user calls `update_prep!`
* `aviary_input_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of input variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary input variables.
* `aviary_output_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of output variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary output variables.
* `aviary_meta_data::Dict{String,Any}`: mapping of Aviary variable names to aviary metadata. Currently only the `"units"` and `"default_value"` fields are used.
"""
function DenseADExplicitComp(ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, aviary_input_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_output_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_meta_data=Dict{String,Nothing}()) where {TAD<:ADTypes.AbstractADType}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (X,)->begin
            return f(X, params)
        end
    end

    # Process the Aviary metadata for the inputs.
    X_ca_full, units_dict_tmp, aviary_input_names = _process_aviary_metadata(X_ca, units_dict, aviary_input_vars, aviary_meta_data)
    # Process the Aviary metadata for the outputs.
    Y_ca = compute_adable(X_ca_full)
    Y_ca_full, units_dict_full, aviary_output_names = _process_aviary_metadata(Y_ca, units_dict_tmp, aviary_output_vars, aviary_meta_data)

    _check_aviary_names(aviary_input_names, aviary_output_names)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        prep, J_ca, rcdict, X_ca_cs = _get_dense_prep_stuff(ad_backend, compute_adable, X_ca_full)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        prep = J_ca = X_ca_cs = nothing
    end

    return DenseADExplicitComp{false}(ad_backend, compute_adable, X_ca_full, J_ca, prep, units_dict_full, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end

function _get_dense_prep_stuff(ad_backend, f!, Y_ca, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f!, Y_ca, ad_backend, X_ca)

    # Get the Jacobian matrix.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    TCS = Complex{TF}
    X_ca_cs = similar(X_ca, TCS)
    Y_ca_cs = similar(Y_ca, TCS)

    return prep, J_ca, X_ca_cs, Y_ca_cs
end

function _get_dense_prep_stuff(ad_backend, f, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f, ad_backend, X_ca)

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = f(X_ca)

    # Now I think I can get the sparse Jacobian from that.
    TF = promote_type(eltype(Y_ca), eltype(X_ca))
    J = Matrix{TF}(undef, length(Y_ca), length(X_ca))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca = ComponentMatrix(J, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return prep, J_ca, X_ca_cs
end

function update_prep(self::DenseADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

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
        f! = get_callback(self)
        prep, J_ca, X_ca_cs, Y_ca_cs = _get_dense_prep_stuff(ad_backend, f!, Y_ca, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict
        aviary_input_names = self.aviary_input_names
        aviary_output_names = self.aviary_output_names
        aviary_meta_data = self.aviary_meta_data

        self = DenseADExplicitComp{true}(ad_backend, f!, X_ca, Y_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    end

    return self
end

function update_prep(self::DenseADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if length(input_sizes) > 0
        X_ca_old = get_input_ca(self)

        X_ca = _resize_component_vector(X_ca_old, input_sizes)

        # Get the new sparsity stuff.
        ad_backend = get_backend(self)
        f = get_callback(self)
        prep, J_ca, X_ca_cs = _get_dense_prep_stuff(ad_backend, f, X_ca)

        # Now just copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict
        aviary_input_names = self.aviary_input_names
        aviary_output_names = self.aviary_output_names
        aviary_meta_data = self.aviary_meta_data

        self = DenseADExplicitComp{false}(ad_backend, f, X_ca, J_ca, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    end

    return self
end

function get_partials_data(self::DenseADExplicitComp)
    return [OpenMDAOCore.PartialsData("*", "*")]
end

function setup_partials(self::DenseADExplicitComp, input_sizes, output_sizes)

    input_av_name_to_ca_name = Dict(get_aviary_input_name(self, k)=>k for k in keys(get_input_ca(self)))
    input_sizes_ca = Dict{Symbol,Any}(input_av_name_to_ca_name[aviary_name]=>sz for (aviary_name, sz) in input_sizes)

    output_av_name_to_ca_name = Dict(get_aviary_output_name(self, k)=>k for k in keys(get_output_ca(self)))
    output_sizes_ca = Dict{Symbol,Any}(output_av_name_to_ca_name[aviary_name]=>sz for (aviary_name, sz) in output_sizes)

    self_new = update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, get_partials_data(self_new)
end

function OpenMDAOCore.compute_partials!(self::DenseADExplicitComp{true}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    Y_ca = get_output_ca(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    raxis, caxis = getaxes(J_ca)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca[oname, iname])

            # Now need to permute the dimensions to account for the Julia vs Python ordering.
            Jsub_in_pd = PermutedDimsArray(Jsub_in, ndims(Jsub_in):-1:1)

            # This gets the underlying Array that stores the current sub-Jacobian that OpenMDAO allocates on the Python side.
            iname_aviary = get_aviary_input_name(self, iname)
            oname_aviary = get_aviary_output_name(self, oname)
            Jsub_out = partials[oname_aviary, iname_aviary]

            # Now we should be able to write the partials to the OpenMDAO Python array.
            Jsub_out .= Jsub_in_pd
        end
    end

    return nothing
end

function OpenMDAOCore.compute_partials!(self::DenseADExplicitComp{false}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Get the Jacobian.
    f = get_callback(self)
    # Y_ca = get_output_ca(self)
    J_ca = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca` and put them in `partials`.
    raxis, caxis = getaxes(J_ca)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca[oname, iname])
            
            # Now need to permute the dimensions to account for the Julia vs Python ordering.
            Jsub_in_pd = PermutedDimsArray(Jsub_in, ndims(Jsub_in):-1:1)

            # This gets the underlying Array that stores the nonzero entries in the current sub-Jacobian that OpenMDAO sees.
            iname_aviary = get_aviary_input_name(self, iname)
            oname_aviary = get_aviary_output_name(self, oname)
            Jsub_out = partials[oname_aviary, iname_aviary]

            # Now we should be able to write the partials to the OpenMDAO Python array.
            Jsub_out .= Jsub_in_pd
        end
    end

    return nothing
end

has_setup_partials(self::DenseADExplicitComp) = true
has_compute_partials(self::DenseADExplicitComp) = true
has_compute_jacvec_product(self::DenseADExplicitComp) = false
