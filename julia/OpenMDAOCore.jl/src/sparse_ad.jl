abstract type AbstractAutoSparseForwardDiffExplicitComp <: AbstractExplicitComp end

get_callback(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.compute_forwarddiffable!

get_input_ca(::Type{Float64}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.X_ca
get_input_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = get_input_ca(Float64, comp)

get_output_ca(::Type{Float64}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.Y_ca
get_output_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = get_output_ca(Float64, comp)

get_sparse_jacobian_ca(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.J_ca_sparse
get_sparse_jacobian_cache(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.jac_cache
get_units(comp::AbstractAutoSparseForwardDiffExplicitComp, varname) = comp.units_dict[varname]
get_rows_cols_dict(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.rcdict

function get_input_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k)) for k in keys(ca)]
end

function get_partials_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    rcdict = get_rows_cols_dict(self)
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]
        # Convert from 1-based to 0-based indexing.
        rows0based = rows .- 1
        cols0based = cols .- 1
        push!(partials_data, OpenMDAOCore.PartialsData(string(output_name), string(input_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end

function OpenMDAOCore.setup(self::AbstractAutoSparseForwardDiffExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)
    partials_data = get_partials_data(self)

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::AbstractAutoSparseForwardDiffExplicitComp, inputs, outputs)
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
        outputs[string(oname)] .= @view(Y_ca[oname])
    end

    return nothing
end

_maybe_nonzeros(A::AbstractArray) = A
_maybe_nonzeros(A::AbstractSparseArray) = nonzeros(A)
_maybe_nonzeros(A::Base.ReshapedArray{T,N,P}) where {T,N,P<:AbstractSparseArray} = nonzeros(parent(A))

function OpenMDAOCore.compute_partials!(self::AbstractAutoSparseForwardDiffExplicitComp, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    J_ca_sparse = get_sparse_jacobian_ca(self)
    jac_cache = get_sparse_jacobian_cache(self)
    forwarddiff_color_jacobian!(J_ca_sparse, f!, X_ca, jac_cache)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    rcdict = get_rows_cols_dict(self)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = J_ca_sparse[oname, iname]
            
            # Need to reshape the subjacobian to correspond to the rows and cols.
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)

            # Grab the entry in partials we're interested in, and write the data we want to it.
            rows, cols = rcdict[oname, iname]

            # This gets the underlying Vector that stores the nonzero entries in the current sub-Jacobian that OpenMDAO sees.
            Jsub_out = partials[string(oname), string(iname)]

            # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
            Jsub_out_vec = _maybe_nonzeros(Jsub_out)

            # Now write the non-zero entries to Jsub_out_vec.
            Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
        end
    end

    return nothing
end

has_setup_partials(self::AbstractAutoSparseForwardDiffExplicitComp) = false
has_compute_partials(self::AbstractAutoSparseForwardDiffExplicitComp) = true
has_compute_jacvec_product(self::AbstractAutoSparseForwardDiffExplicitComp) = false

function generate_perturbed_jacobian!(J_ca, f!, Y_ca, X_ca, nevals=3, rel_perturb=0.001)
    if nevals < 1
        raise(ArgumentError("nevals must be >=1, but is $nevals"))
    end
    X_perturb = similar(X_ca)
    Y_perturb = similar(Y_ca)
    J_tmp = similar(J_ca)
    perturb = similar(X_ca)

    J_ca .= 0.0
    for i in 1:nevals
        rand!(perturb)
        X_perturb .= (1 .+ rel_perturb.*perturb).*X_ca
        ForwardDiff.jacobian!(J_tmp, f!, Y_perturb, X_perturb)
        J_ca .+= abs.(J_tmp./nevals)
    end
    
    return nothing
end
