struct MatrixFreeADExplicitComp{InPlace,TAD,TCompute,TX,TY,TdY,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    dX_ca::TX
    dY_ca::TdY
    prep::TPrep
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS

    function MatrixFreeADExplicitComp{true}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
        return new{true,typeof(ad_backend),typeof(compute_adable),typeof(X_ca),typeof(Y_ca),typeof(dY_ca),typeof(prep),typeof(X_ca_cs),typeof(Y_ca_cs)}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
    end

    function MatrixFreeADExplicitComp{false}(ad_backend, compute_adable, X_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs)
        Y_ca = Y_ca_cs = nothing
        return new{false,typeof(ad_backend),typeof(compute_adable),typeof(X_ca),typeof(Y_ca),typeof(dY_ca),typeof(prep),typeof(X_ca_cs),typeof(Y_ca_cs)}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
    end
end

get_dinput_ca(comp::MatrixFreeADExplicitComp) = comp.dX_ca
get_doutput_ca(comp::MatrixFreeADExplicitComp) = comp.dY_ca

function get_partials_data(self::MatrixFreeADExplicitComp)
    return Vector{PartialsData}()
end

"""
    MatrixFreeADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_mode=nothing, disable_prep=false)

Create a `MatrixFreeADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AbstractADType` automatic differentation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `force_mode=nothing`: If `fwd`, use `DifferentiationInterface.pushforward!` to compute the derivatives (aka perform a Jacobian-vector product).
  If `rev`, use `DifferentiationInterface.pullback!` to compute the derivatives (aka perform a vector-Jacobian product).
  If `nothing`, use whatever would be faster, as determined by `DifferentiationInterface.pushforward_performance` and `DifferentiationInterface.pullback_performance`, prefering `pushforward.
* `disable_prep`: if `true`, do not use either `prepare_pushforward` or `prepare_pullback` to create a `DifferentiationInterface.PushforwardPrep` or `PullbackPrep` object to accelerate the derivative calculation.
  Disabling prep can avoid correctness issues with ReverseDiff.jl, see [the discussion of branching and the `AbstractTape` API](https://juliadiff.org/ReverseDiff.jl/dev/api/#The-AbstractTape-API).
"""
function MatrixFreeADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_mode=nothing, disable_prep=false)
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # Create copies of X_ca and Y_ca that will be use for the input and output tangents (not sure if that's the correct terminology).
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)

    # Get the "preparation" objective for efficiency.
    if force_mode === nothing
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            if disable_prep
                prep = DifferentiationInterface.NoPushforwardPrep()
            else
                prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
            end
        else
            if disable_prep
                prep = DifferentiationInterface.NoPullbackPrep()
            else
                prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
            end
        end
    elseif force_mode == "fwd"
        if disable_prep
            prep = DifferentiationInterface.NoPushforwardPrep()
        else
            prep = DifferentiationInterface.prepare_pushforward(compute_adable, Y_ca, ad_backend, X_ca, (dX_ca,))
        end
    elseif force_mode == "rev"
        if disable_prep
            prep = DifferentiationInterface.NoPullbackPrep()
        else
            prep = DifferentiationInterface.prepare_pullback(compute_adable, Y_ca, ad_backend, X_ca, (dY_ca,))
        end
    else
        throw(ArgumentError("force_mode keyword argument should be one of `nothing`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return MatrixFreeADExplicitComp{true}(ad_backend, compute_adable, X_ca, Y_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
end

function MatrixFreeADExplicitComp(ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), force_mode=nothing, disable_prep=false)
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (X,)->begin
            return f(X, params)
        end
    end

    # Create copies of X_ca and Y_ca that will be use for the input and output tangents (not sure if that's the correct terminology).
    dX_ca = similar(X_ca)
    Y_ca = compute_adable(X_ca)
    dY_ca = similar(Y_ca)

    # Get the "preparation" objective for efficiency.
    if force_mode === nothing
        if DifferentiationInterface.pushforward_performance(ad_backend) isa DifferentiationInterface.PushforwardFast
            if disable_prep
                prep = DifferentiationInterface.NoPushforwardPrep()
            else
                prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
            end
        else
            if disable_prep
                prep = DifferentiationInterface.NoPullbackPrep()
            else
                prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
            end
        end
    elseif force_mode == "fwd"
        if disable_prep
            prep = DifferentiationInterface.NoPushforwardPrep()
        else
            prep = DifferentiationInterface.prepare_pushforward(compute_adable, ad_backend, X_ca, (dX_ca,))
        end
    elseif force_mode == "rev"
        if disable_prep
            prep = DifferentiationInterface.NoPullbackPrep()
        else
            prep = DifferentiationInterface.prepare_pullback(compute_adable, ad_backend, X_ca, (dY_ca,))
        end
    else
        throw(ArgumentError("force_mode keyword argument should be one of `nothing`, \"fwd\", or \"rev\" but is $(force_mode)"))
    end

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    # Y_ca_cs = similar(Y_ca, ComplexF64)

    return MatrixFreeADExplicitComp{false}(ad_backend, compute_adable, X_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs)
end

function _compute_pushforward!(self::MatrixFreeADExplicitComp{true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
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
    for iname in keys(d_inputs)
        @view(dX_ca[Symbol(iname)]) .= d_inputs[iname]
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
        d_outputs[String(oname)] .= @view(dY_ca[oname])
    end

    return nothing
end

function _compute_pushforward!(self::MatrixFreeADExplicitComp{false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
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
    for iname in keys(d_inputs)
        @view(dX_ca[Symbol(iname)]) .= d_inputs[iname]
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
        d_outputs[String(oname)] .= @view(dY_ca[oname])
    end

    return nothing
end

function _compute_pullback!(self::MatrixFreeADExplicitComp{true}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    dY_ca = get_doutput_ca(self)
    dY_ca .= 0.0
    for oname in keys(d_outputs)
        @view(dY_ca[Symbol(oname)]) .= d_outputs[oname]
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
        d_inputs[String(iname)] .= @view(dX_ca[iname])
    end

    return nothing
end

function _compute_pullback!(self::MatrixFreeADExplicitComp{false}, inputs, d_inputs, d_outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    dY_ca = get_doutput_ca(self)
    dY_ca .= 0.0
    for oname in keys(d_outputs)
        @view(dY_ca[Symbol(oname)]) .= d_outputs[oname]
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
        d_inputs[String(iname)] .= @view(dX_ca[iname])
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
            throw(ArgumentError("mode = \"fwd\" not supported for AD backend $(backend), preparation $(prep)"))
        end
    elseif mode == "rev"
        if prep isa DifferentiationInterface.PullbackPrep
            _compute_pullback!(self, inputs, d_inputs, d_outputs)
        else
            throw(ArgumentError("mode = \"rev\" not supported for AD backend $(backend), preparation $(prep)"))
        end
    else
        throw(ArgumentError("unknown mode = \"$(mode)\""))
    end

    return nothing
end
