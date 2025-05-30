struct MatrixFreeADExplicitComp{InPlace,TAD,TCompute,TX,TY,TdX,TdY,TPrep,TXCS,TYCS,TAMD} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    dX_ca::TdX
    dY_ca::TdY
    force_mode::String
    disable_prep::Bool
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

    function MatrixFreeADExplicitComp{InPlace}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data) where {InPlace}
        return new{InPlace,typeof(ad_backend), typeof(compute_adable), typeof(X_ca), typeof(Y_ca), typeof(dX_ca), typeof(dY_ca), typeof(prep), typeof(X_ca_cs), typeof(Y_ca_cs), typeof(aviary_meta_data)}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    end
end

function MatrixFreeADExplicitComp{false}(ad_backend, compute_adable, X_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    Y_ca = nothing
    Y_ca_cs = nothing
    return MatrixFreeADExplicitComp{false}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end
get_dinput_ca(comp::MatrixFreeADExplicitComp) = comp.dX_ca
get_doutput_ca(comp::MatrixFreeADExplicitComp) = comp.dY_ca
get_force_mode(comp::MatrixFreeADExplicitComp) = comp.force_mode
get_disable_prep(comp::MatrixFreeADExplicitComp) = comp.disable_prep

function get_partials_data(self::MatrixFreeADExplicitComp)
    return Vector{PartialsData}()
end

"""
    MatrixFreeADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_mode="", disable_prep=false, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), aviary_input_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_output_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_meta_data=Dict{String,Any}())

Create a `MatrixFreeADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `force_mode=""`: If `"fwd"`, use `DifferentiationInterface.pushforward!` to compute the derivatives (aka perform a Jacobian-vector product).
  If `"rev"`, use `DifferentiationInterface.pullback!` to compute the derivatives (aka perform a vector-Jacobian product).
  If `""`, use whatever would be faster, as determined by `DifferentiationInterface.pushforward_performance` and `DifferentiationInterface.pullback_performance`, prefering `pushforward`.
* `disable_prep`: if `true`, do not use either `prepare_pushforward` or `prepare_pullback` to create a `DifferentiationInterface.PushforwardPrep` or `PullbackPrep` object to accelerate the derivative calculation.
  Disabling prep can avoid correctness issues with ReverseDiff.jl, see [the discussion of branching and the `AbstractTape` API](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API).
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `aviary_input_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of input variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary input variables.
* `aviary_output_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of output variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary output variables.
* `aviary_meta_data::Dict{String,Any}`: mapping of Aviary variable names to aviary metadata. Currently only the `"units"` and `"default_value"` fields are used.
"""
function MatrixFreeADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, force_mode="", disable_prep=false, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), aviary_input_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_output_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_meta_data=Dict{String,Nothing}())

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

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        prep, dX_ca, dY_ca, X_ca_cs, Y_ca_cs = _get_matrix_free_prep_stuff_in_place(ad_backend, compute_adable, Y_ca_full, X_ca_full, force_mode, disable_prep)
    else
        dX_ca = ComponentVector{eltype(X_ca_full)}()
        dY_ca = ComponentVector{eltype(Y_ca_full)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        Y_ca_cs = ComponentVector{ComplexF64}()
        # Doesn't matter if we chose NoPushforwardPrep() or NoPullbackPrep(), since it will be set to the correct thing later in `update_prep`.
        prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, Y_ca_full, ad_backend, X_ca_full, (dX_ca,); strict=Val{true}()))
    end

    return MatrixFreeADExplicitComp{true}(ad_backend, compute_adable, X_ca_full, Y_ca_full, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict_full, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end

"""
    MatrixFreeADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, force_mode="", disable_prep=false, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), aviary_input_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_output_vars=Dict{Symbol,Dict{String,<:Any}}(), aviary_meta_data=Dict{String,Any}())

Create a `MatrixFreeADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f`: function of the form `Y_ca = f(X_ca, params)` which returns outputs `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `force_mode=""`: If `"fwd"`, use `DifferentiationInterface.pushforward!` to compute the derivatives (aka perform a Jacobian-vector product).
  If `"rev"`, use `DifferentiationInterface.pullback!` to compute the derivatives (aka perform a vector-Jacobian product).
  If `""`, use whatever would be faster, as determined by `DifferentiationInterface.pushforward_performance` and `DifferentiationInterface.pullback_performance`, prefering `pushforward`.
* `disable_prep`: if `true`, do not use either `prepare_pushforward` or `prepare_pullback` to create a `DifferentiationInterface.PushforwardPrep` or `PullbackPrep` object to accelerate the derivative calculation.
  Disabling prep can avoid correctness issues with ReverseDiff.jl, see [the discussion of branching and the `AbstractTape` API](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API).
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `shape_by_conn_dict`: `Dict` mapping variable names (as `Symbol`s) to `Bool`s indicating if the variable's shape (size) will be set dynamically by a connection
* `copy_shape_dict`: `Dict` mapping variable names to other variable names indicating the "key" symbol should take its size from the "value" symbol
* `aviary_input_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of input variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary input variables.
* `aviary_output_vars::Dict{Symbol,Dict{String,<:Any}}`: mapping of output variable names to a `Dict` that contains keys `name` and optionally `shape` defining the Aviary name and shape for Aviary output variables.
* `aviary_meta_data::Dict{String,Any}`: mapping of Aviary variable names to aviary metadata. Currently only the `"units"` and `"default_value"` fields are used.
"""
function MatrixFreeADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, force_mode="", disable_prep=false, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), aviary_input_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_output_vars=Dict{Symbol,Dict{String,Nothing}}(), aviary_meta_data=Dict{String,Nothing}())

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

    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0)
        prep, dX_ca, dY_ca, X_ca_cs = _get_matrix_free_prep_stuff_out_of_place(ad_backend, compute_adable, Y_ca_full, X_ca_full, force_mode, disable_prep)
    else
        dX_ca = ComponentVector{eltype(X_ca_full)}()
        dY_ca = ComponentVector{eltype(Y_ca_full)}()
        X_ca_cs = ComponentVector{ComplexF64}()
        # Doesn't matter if we chose NoPushforwardPrep() or NoPullbackPrep(), since it will be set to the correct thing later in `update_prep`.
        prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca_full, (dX_ca,); strict=Val{true}()))
    end

    return MatrixFreeADExplicitComp{false}(ad_backend, compute_adable, X_ca_full, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict_full, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
end

function _get_matrix_free_prep_stuff_in_place(ad_backend, compute_adable, Y_ca, X_ca, force_mode::String, disable_prep::Bool)
    # Create copies of X_ca and Y_ca that will be use for the input and output tangents (not sure if that's the correct terminology).
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)

    # Get the "preparation" objective for efficiency.
    if force_mode == ""
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            if disable_prep
                prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,); strict=Val{true}()))
            else
                prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
            end
        else
            if disable_prep
                prep = DifferentiationInterface.NoPullbackPrep(DifferentiationInterface.signature(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,); strict=Val{true}()))
            else
                prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
            end
        end
    elseif force_mode == "fwd"
        if disable_prep
            prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,); strict=Val{true}()))
        else
            prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
        end
    elseif force_mode == "rev"
        if disable_prep
            prep = DifferentiationInterface.NoPullbackPrep(DifferentiationInterface.signature(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,); strict=Val{true}()))
        else
            prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
        end
    else
        throw(ArgumentError("force_mode argument should be one of `\"\"`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end

    # Create complex-valued versions of the X_ca_full and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return prep, dX_ca, dY_ca, X_ca_cs, Y_ca_cs
end

function _get_matrix_free_prep_stuff_out_of_place(ad_backend, compute_adable, Y_ca, X_ca, force_mode::String, disable_prep::Bool)
    # Create a copy of X_ca that will be use for the input tangents (not sure if that's the correct terminology).
    dX_ca = similar(X_ca)

    # Create a copy of Y_ca_full that will be use for the output tangents (not sure if that's the correct terminology).
    dY_ca = similar(Y_ca)

    # Get the "preparation" objective for efficiency.
    if force_mode == ""
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            if disable_prep
                prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca, (dX_ca,); strict=Val{true}()))
            else
                prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
            end
        else
            if disable_prep
                prep = DifferentiationInterface.NoPullbackPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca, (dY_ca,); strict=Val{true}()))
            else
                prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
            end
        end
    elseif force_mode == "fwd"
        if disable_prep
            prep = DifferentiationInterface.NoPushforwardPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca, (dX_ca,); strict=Val{true}()))
        else
            prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
        end
    elseif force_mode == "rev"
        if disable_prep
            prep = DifferentiationInterface.NoPullbackPrep(DifferentiationInterface.signature(compute_adable, ad_backend, X_ca, (dY_ca,); strict=Val{true}()))
        else
            prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
        end
    else
        throw(ArgumentError("force_mode argument should be one of `\"\"`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return prep, dX_ca, dY_ca, X_ca_cs
end

# function update_prep!(self::MatrixFreeADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
#     if (length(input_sizes) > 0) || (length(output_sizes) > 0)
#         X_ca_old = get_input_ca(self)
#         Y_ca_old = get_output_ca(self)
#
#         # Create a new versions of `X_ca_old` and `Y_ca_old` that have the correct sizes and default values.
#         X_ca = _resize_component_vector(X_ca_old, input_sizes)
#         Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
#
#         # Get the new prep stuff.
#         prep, dX_ca, dY_ca, X_ca_cs, Y_ca_cs = _get_matrix_free_prep_stuff_in_place(get_backend(self), get_callback(self), Y_ca, X_ca, get_force_mode(self), get_disable_prep(self))
#
#         # Save everything in this struct.
#         self.X_ca = X_ca
#         self.Y_ca = Y_ca
#         self.dX_ca = dX_ca
#         self.dY_ca = dY_ca
#         self.prep = prep
#         self.X_ca_cs = X_ca_cs
#         self.Y_ca_cs = Y_ca_cs
#     end
#
#     return nothing
# end

function update_prep(self::MatrixFreeADExplicitComp{true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        # Create a new versions of `X_ca_old` and `Y_ca_old` that have the correct sizes and default values.
        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        # Get the new prep stuff.
        ad_backend = get_backend(self)
        f! = get_callback(self)
        force_mode = get_force_mode(self)
        disable_prep = get_disable_prep(self)
        prep, dX_ca, dY_ca, X_ca_cs, Y_ca_cs = _get_matrix_free_prep_stuff_in_place(ad_backend, f!, Y_ca, X_ca, get_force_mode(self), get_disable_prep(self))

        # Copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict
        aviary_input_names = self.aviary_input_names
        aviary_output_names = self.aviary_output_names
        aviary_meta_data = self.aviary_meta_data
        self = MatrixFreeADExplicitComp{true}(ad_backend, f!, X_ca, Y_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    end

    return self
end

# function update_prep!(self::MatrixFreeADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
#     if (length(input_sizes) > 0) || (length(output_sizes) > 0)
#         X_ca_old = get_input_ca(self)
#         Y_ca_old = get_output_ca(self)
#
#         # Create a new versions of `X_ca_old` and `Y_ca_old` that have the correct sizes and default values.
#         X_ca = _resize_component_vector(X_ca_old, input_sizes)
#         Y_ca = _resize_component_vector(Y_ca_old, output_sizes)
#
#         # Get the new prep stuff.
#         prep, dX_ca, dY_ca, X_ca_cs = _get_matrix_free_prep_stuff_out_of_place(get_backend(self), get_callback(self), Y_ca, X_ca, get_force_mode(self), get_disable_prep(self))
#
#         # Save everything in this struct.
#         self.X_ca = X_ca
#         self.Y_ca = Y_ca
#         self.dX_ca = dX_ca
#         self.dY_ca = dY_ca
#         self.prep = prep
#         self.X_ca_cs = X_ca_cs
#     end
#
#     return nothing
# end

function update_prep(self::MatrixFreeADExplicitComp{false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = get_input_ca(self)
        Y_ca_old = get_output_ca(self)

        # Create a new versions of `X_ca_old` and `Y_ca_old` that have the correct sizes and default values.
        X_ca = _resize_component_vector(X_ca_old, input_sizes)
        Y_ca = _resize_component_vector(Y_ca_old, output_sizes)

        # Get the new prep stuff.
        ad_backend = get_backend(self)
        f = get_callback(self)
        force_mode = get_force_mode(self)
        disable_prep = get_disable_prep(self)
        prep, dX_ca, dY_ca, X_ca_cs = _get_matrix_free_prep_stuff_out_of_place(ad_backend, f, Y_ca, X_ca, force_mode, disable_prep)

        # Copy things over.
        units_dict = self.units_dict
        tags_dict = self.tags_dict
        shape_by_conn_dict = self.shape_by_conn_dict
        copy_shape_dict = self.copy_shape_dict
        aviary_input_names = self.aviary_input_names
        aviary_output_names = self.aviary_output_names
        aviary_meta_data = self.aviary_meta_data
        self = MatrixFreeADExplicitComp{false}(ad_backend, f, X_ca, dX_ca, dY_ca, force_mode, disable_prep, prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, aviary_input_names, aviary_output_names, aviary_meta_data)
    end

    return self
end

function setup_partials(self::MatrixFreeADExplicitComp, input_sizes, output_sizes)
    input_av_name_to_ca_name = Dict(get_aviary_input_name(self, k)=>k for k in keys(get_input_ca(self)))
    input_sizes_ca = Dict{Symbol,Any}(input_av_name_to_ca_name[aviary_name]=>sz for (aviary_name, sz) in input_sizes)

    output_av_name_to_ca_name = Dict(get_aviary_output_name(self, k)=>k for k in keys(get_output_ca(self)))
    output_sizes_ca = Dict{Symbol,Any}(output_av_name_to_ca_name[aviary_name]=>sz for (aviary_name, sz) in output_sizes)

    self_new = update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, get_partials_data(self_new)
end

function _compute_pushforward!(self::MatrixFreeADExplicitComp{true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Hmm... how do the sizes work?
    # Well, I think d_inputs has the derivative each input wrt some upstream input (a design variable, I guess).
    # So shouldn't that be the same size as the input vector itself?
    # Well, is it possible that it wouldn't include all of them?
    # I guess it could be, right?
    # Maybe one of the inputs to this particular component is not impacted by the design variables.
    # So I guess I should iterate over the keys in d_inputs.
    # And I should also zero out the input vector, in case d_inputs doesn't have everything.
    dX_ca = get_dinput_ca(self)
    dX_ca .= 0.0
    for iname in keys(dX_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        if iname_aviary in keys(d_inputs)
            @view(dX_ca[iname]) .= d_inputs[iname_aviary]
        end
    end

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Get the AD backend.
    backend = get_backend(self)

    # Now actually do the Jacobian-vector product.
    DifferentiationInterface.pushforward!(compute_adable, Y_ca, (dY_ca,), prep, backend, X_ca, (dX_ca,))

    # Now copy the output derivatives to `d_outputs`:
    for oname in keys(dY_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        if typeof(d_outputs[oname_aviary]) <: AbstractArray
            d_outputs[oname_aviary] .= @view(dY_ca[oname])
        else
            d_outputs[oname_aviary] = only(dY_ca[oname])
        end
    end

    return nothing
end

function _compute_pushforward!(self::MatrixFreeADExplicitComp{false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Hmm... how do the sizes work?
    # Well, I think d_inputs has the derivative each input wrt some upstream input (a design variable, I guess).
    # So shouldn't that be the same size as the input vector itself?
    # Well, is it possible that it wouldn't include all of them?
    # I guess it could be, right?
    # Maybe one of the inputs to this particular component is not impacted by the design variables.
    # So I guess I should iterate over the keys in d_inputs.
    # And I should also zero out the input vector, in case d_inputs doesn't have everything.
    dX_ca = get_dinput_ca(self)
    dX_ca .= 0.0
    for iname in keys(dX_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        if iname_aviary in keys(d_inputs)
            @view(dX_ca[iname]) .= d_inputs[iname_aviary]
        end
    end

    # We'll write the result of the Jacobian-vector product to a different component array.
    dY_ca = get_doutput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Get the AD backend.
    backend = get_backend(self)

    # Now actually do the Jacobian-vector product.
    DifferentiationInterface.pushforward!(compute_adable, (dY_ca,), prep, backend, X_ca, (dX_ca,))

    # Now copy the output derivatives to `d_outputs`:
    for oname in keys(dY_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        if typeof(d_outputs[oname_aviary]) <: AbstractArray
            d_outputs[oname_aviary] .= @view(dY_ca[oname])
        else
            d_outputs[oname_aviary] = only(dY_ca[oname])
        end
    end

    return nothing
end

function _compute_pullback!(self::MatrixFreeADExplicitComp{true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    dY_ca = get_doutput_ca(self)
    dY_ca .= 0.0
    for oname in keys(dY_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        if oname_aviary in keys(d_outputs)
            @view(dY_ca[oname]) .= d_outputs[oname_aviary]
        end
    end

    # The AD library will need the output component array to do the Jacobian-vector product.
    Y_ca = get_output_ca(self)

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Need the AD backend.
    backend = get_backend(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Now actually do the Jacobian-vector product.
    DifferentiationInterface.pullback!(compute_adable, Y_ca, (dX_ca,), prep, backend, X_ca, (dY_ca,))

    # Now copy the input derivatives to `d_inputs`:
    for iname in keys(dX_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        if typeof(d_inputs[iname_aviary]) <: AbstractArray
            d_inputs[iname_aviary] .= @view(dX_ca[iname])
        else
            d_inputs[iname_aviary] = only(dX_ca[iname])
        end
    end

    return nothing
end

function _compute_pullback!(self::MatrixFreeADExplicitComp{false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    dY_ca = get_doutput_ca(self)
    dY_ca .= 0.0
    for oname in keys(dY_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        if oname_aviary in keys(d_outputs)
            @view(dY_ca[oname]) .= d_outputs[oname_aviary]
        end
    end

    # We'll write the result of the vector-Jacobian product to a different component array.
    dX_ca = get_dinput_ca(self)

    # This is the function that will actually do the computation.
    compute_adable = get_callback(self)

    # Need the AD backend.
    backend = get_backend(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # Now actually do the Jacobian-vector product.
    DifferentiationInterface.pullback!(compute_adable, (dX_ca,), prep, backend, X_ca, (dY_ca,))

    # Now copy the input derivatives to `d_inputs`:
    for iname in keys(dX_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        if typeof(d_inputs[iname_aviary]) <: AbstractArray
            d_inputs[iname_aviary] .= @view(dX_ca[iname])
        else
            d_inputs[iname_aviary] = only(dX_ca[iname])
        end
    end

    return nothing
end

function OpenMDAOCore.compute_jacvec_product!(self::MatrixFreeADExplicitComp, inputs, d_inputs, d_outputs, mode)
    backend = get_backend(self)
    prep = get_prep(self)
    if mode == "fwd"
        if prep isa DifferentiationInterface.PushforwardPrep
            _compute_pushforward!(self, inputs, d_inputs, d_outputs)
        else
            @warn "mode = \"fwd\" not supported for AD backend $(backend), preparation $(prep), derivatives for $(self) will be incorrect"
        end
    elseif mode == "rev"
        if prep isa DifferentiationInterface.PullbackPrep
            _compute_pullback!(self, inputs, d_inputs, d_outputs)
        else
            @warn "mode = \"rev\" not supported for AD backend $(backend), preparation $(prep), derivatives for $(self) will be incorrect"
        end
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end

    return nothing
end

