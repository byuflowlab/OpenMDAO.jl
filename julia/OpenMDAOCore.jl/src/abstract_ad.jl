abstract type AbstractADExplicitComp{InPlace} <: AbstractExplicitComp end

get_callback(comp::AbstractADExplicitComp) = comp.compute_adable

get_input_ca(::Type{Float64}, comp::AbstractADExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::AbstractADExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::AbstractADExplicitComp) = comp.X_ca
get_input_ca(comp::AbstractADExplicitComp) = get_input_ca(Float64, comp)

get_output_ca(::Type{Float64}, comp::AbstractADExplicitComp{true}) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::AbstractADExplicitComp{true}) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::AbstractADExplicitComp{true}) = comp.Y_ca

get_output_ca(::Type{Float64}, comp::AbstractADExplicitComp{false}) = get_callback(comp)(get_input_ca(Float64, comp))
get_output_ca(::Type{ComplexF64}, comp::AbstractADExplicitComp{false}) = get_callback(comp)(get_input_ca(ComplexF64, comp))
get_output_ca(::Type{Any}, comp::AbstractADExplicitComp{false}) = get_callback(comp)(get_input_ca(Float64, comp))

get_output_ca(comp::AbstractADExplicitComp) = get_output_ca(Float64, comp)

get_prep(comp::AbstractADExplicitComp) = comp.prep
get_units(comp::AbstractADExplicitComp, varname) = get(comp.units_dict, varname, "unitless")
get_tags(comp::AbstractADExplicitComp, varname) = get(comp.tags_dict, varname, Vector{String}())
get_backend(comp::AbstractADExplicitComp) = comp.ad_backend

get_aviary_input_name(comp::AbstractADExplicitComp, ca_name::Symbol) = get(comp.aviary_input_names, ca_name, string(ca_name))
get_aviary_input_name(comp::AbstractADExplicitComp, ca_name::AbstractString) = get_aviary_input_name(comp, Symbol(ca_name))
get_aviary_input_name(comp::AbstractADExplicitComp, ca_name::Nothing) = nothing

get_aviary_output_name(comp::AbstractADExplicitComp, ca_name::Symbol) = get(comp.aviary_output_names, ca_name, string(ca_name))
get_aviary_output_name(comp::AbstractADExplicitComp, ca_name::AbstractString) = get_aviary_input_name(comp, Symbol(ca_name))
get_aviary_output_name(comp::AbstractADExplicitComp, ca_name::Nothing) = nothing

function get_aviary_name(comp::AbstractADExplicitComp, ca_name::Symbol)
    if ca_name in keys(comp.aviary_input_names)
        name = get_aviary_input_name(comp, ca_name)
    elseif ca_name in keys(comp.aviary_output_names)
        name = get_aviary_output_name(comp, ca_name)
    else
        name = string(ca_name)
    end
    return name
end
get_aviary_name(comp::AbstractADExplicitComp, ca_name::AbstractString) = get_aviary_name(comp, Symbol(ca_name))
get_aviary_name(comp::AbstractADExplicitComp, ca_name::Nothing) = nothing

function get_input_var_data(self::AbstractADExplicitComp)
    ca = get_input_ca(self)
    return [VarData(get_aviary_input_name(self, k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_aviary_name(self, get(self.copy_shape_dict, k, nothing))) for k in keys(ca)]
end

function get_output_var_data(self::AbstractADExplicitComp)
    ca = get_output_ca(self)
    return [VarData(get_aviary_output_name(self, k);
                    shape=size(ca[k]),
                    val=ca[k],
                    units=get_units(self, k),
                    tags=get_tags(self, k),
                    shape_by_conn=get(self.shape_by_conn_dict, k, false),
                    copy_shape=get_aviary_name(self, get(self.copy_shape_dict, k, nothing))) for k in keys(ca)]
end

function OpenMDAOCore.setup(self::AbstractADExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)

    return input_data, output_data, Vector{PartialsData}()
end

function OpenMDAOCore.compute!(self::AbstractADExplicitComp{true}, inputs, outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(eltype(valtype(inputs)), self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Call the actual function.
    Y_ca = get_output_ca(eltype(valtype(outputs)), self)
    f! = get_callback(self)
    f!(Y_ca, X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        # This requires that each output is at least a vector.
        if typeof(outputs[oname_aviary]) <: AbstractArray
            outputs[oname_aviary] .= @view(Y_ca[oname])
        else
            outputs[oname_aviary] = only(Y_ca[oname])
        end
    end

    return nothing
end

function OpenMDAOCore.compute!(self::AbstractADExplicitComp{false}, inputs, outputs)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(eltype(valtype(inputs)), self)
    for iname in keys(X_ca)
        iname_aviary = get_aviary_input_name(self, iname)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[iname_aviary]
    end

    # Call the actual function.
    f = get_callback(self)
    Y_ca = f(X_ca)

    # Copy the output `ComponentArray` to the outputs.
    for oname in keys(Y_ca)
        oname_aviary = get_aviary_output_name(self, oname)
        # This requires that each output is at least a vector.
        if typeof(outputs[oname_aviary]) <: AbstractArray
            outputs[oname_aviary] .= @view(Y_ca[oname])
        else
            outputs[oname_aviary] = only(Y_ca[oname])
        end
    end

    return nothing
end
