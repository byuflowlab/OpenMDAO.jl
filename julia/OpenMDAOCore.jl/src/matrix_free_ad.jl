struct SimpleMatrixFreeForwardDiffExplicitComp{TCompute,TX,TY,TPrep,TXCS,TYCS} <: AbstractExplicitComp
    compute_forwarddiffable!::TCompute
    X_ca::TX
    Y_ca::TY
    dX_ca::TX
    dY_ca::TY
    prep::TPrep
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS
end

get_callback(comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.compute_forwarddiffable!

get_input_ca(::Type{Float64}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.X_ca
get_input_ca(comp::SimpleMatrixFreeForwardDiffExplicitComp) = get_input_ca(Float64, comp)

get_dinput_ca(comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.dX_ca

get_output_ca(::Type{Float64}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.Y_ca
get_output_ca(comp::SimpleMatrixFreeForwardDiffExplicitComp) = get_output_ca(Float64, comp)

get_doutput_ca(comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.dY_ca

get_prep(comp::SimpleMatrixFreeForwardDiffExplicitComp) = comp.prep

get_units(comp::SimpleMatrixFreeForwardDiffExplicitComp, varname) = get(comp.units_dict, varname, nothing)
get_tags(comp::SimpleMatrixFreeForwardDiffExplicitComp, varname) = get(comp.tags_dict, varname, nothing)

function get_input_var_data(self::SimpleMatrixFreeForwardDiffExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::SimpleMatrixFreeForwardDiffExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
end

function get_partials_data(self::SimpleMatrixFreeForwardDiffExplicitComp)
    return Vector{PartialsData}()
end

function OpenMDAOCore.setup(self::SimpleMatrixFreeForwardDiffExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)
    partials_data = get_partials_data(self)

    return input_data, output_data, partials_data
end

"""
    SimpleMatrixFreeForwardDiffExplicitComp(f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}())

Create a `SimpleAutoSparseForwardDiffExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
"""
function SimpleMatrixFreeForwardDiffExplicitComp(f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}())
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_forwarddiffable! = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # Create copies of X_ca and Y_ca that will be use for the input and output tangents (not sure if that's the correct terminology).
    dX_ca = similar(X_ca)
    dY_ca = similar(Y_ca)

    # We'll use ForwardDiff.jl to do the AD.
    backend = DifferentiationInterface.AutoForwardDiff()

    # Get the "preparation" objective for efficiency.
    prep = DifferentiationInterface.prepare_pushforward(compute_forwarddiffable!, Y_ca, backend, X_ca, (dX_ca,))

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return SimpleMatrixFreeForwardDiffExplicitComp(compute_forwarddiffable!, X_ca, Y_ca, dX_ca, dY_ca, prep, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
end

function OpenMDAOCore.compute!(self::SimpleMatrixFreeForwardDiffExplicitComp, inputs, outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(eltype(valtype(inputs)), self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Call the actual function.
    Y_ca = get_output_ca(eltype(valtype(outputs)), self)
    f! = get_callback(self)
    f!(Y_ca, X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        # This requires that each output is at least a vector.
        outputs[string(oname)] .= @view(Y_ca[oname])
    end

    return nothing
end

function OpenMDAOCore.compute_jacvec_product!(self::SimpleMatrixFreeForwardDiffExplicitComp, inputs, d_inputs, d_outputs, mode)
    if mode != "fwd"
        throw(MethodError("only mode = \"fwd\" supported for SimpleMatrixFreeForwardDiffExplicitComp, but passed mode = $(mode)"))
    end

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
    compute_forwarddiffable! = get_callback(self)

    # We stored the "preparation" for reusing.
    prep = get_prep(self)

    # We'll use ForwardDiff.jl to do the AD.
    backend = DifferentiationInterface.AutoForwardDiff()

    # Now actually do the Jacobian-vector product.
    DifferentiationInterface.pushforward!(compute_forwarddiffable!, Y_ca, (dY_ca,), prep, backend, X_ca, (dX_ca,))

    # Now copy the output derivatives to `d_outputs`:
    for oname in keys(dY_ca)
        d_outputs[String(oname)] .= @view(dY_ca[oname])
    end

    return nothing
end
