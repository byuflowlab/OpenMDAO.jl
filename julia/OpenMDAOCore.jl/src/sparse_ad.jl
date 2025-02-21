"""
    SparseADExplicitComp{TAD,TCompute,TX,TY,TJ,TPrep,TRCDict,TXCS,TYCS} <: AbstractExplicitComp

An `<:AbstractADExplicitComp` for sparse Jacobians.

# Fields
* `ad_backend::TAD`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `compute_adable::TCompute`: function of the form `compute_adable(Y, X)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::TX`: `ComponentVector` of inputs
* `Y_ca::TY`: `ComponentVector` of outputs
* `J_ca_sparse::TJ`: Sparse `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `prep::TPrep`: `DifferentiationInterface.jl` "preparation" object
* `rcdict::TRCDict`: `Dict{Tuple{Symbol,Sympol}, Tuple{Vector{Int}, Vector{Int}}` mapping sub-Jacobians of the form `(:output_name, :input_name)` to `Vector`s of non-zero row and column indices (1-based)
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `X_ca::TXCS`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::TXCS`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct SparseADExplicitComp{InPlace,TAD,TCompute,TX,TY,TJ,TPrep,TXCS,TYCS} <: AbstractADExplicitComp{InPlace}
    ad_backend::TAD
    compute_adable::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS
    aviary_input_names::Dict{Symbol,String}
    aviary_output_names::Dict{Symbol,String}

    function SparseADExplicitComp{true}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names)
        return new{true,typeof(ad_backend),typeof(compute_adable),typeof(X_ca),typeof(Y_ca),typeof(J_ca_sparse),typeof(prep),typeof(X_ca_cs),typeof(Y_ca_cs)}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names)
    end

    function SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, X_ca_cs, aviary_input_names, aviary_output_names)
        Y_ca = Y_ca_cs = nothing
        return new{false,typeof(ad_backend),typeof(compute_adable),typeof(X_ca),typeof(Y_ca),typeof(J_ca_sparse),typeof(prep),typeof(X_ca_cs),typeof(Y_ca_cs)}(ad_backend, compute_adable, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names)
    end
end

"""
    SparseADExplicitComp(ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}())

Create a `SparseADExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `ad_backend`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
"""
function SparseADExplicitComp(ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), aviary_input_names=Dict{Symbol,String}(), aviary_output_names=Dict{Symbol,String}(), aviary_meta_data=Dict{String,Any}()) where {TAD<:ADTypes.AutoSparse}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # Process the Aviary metadata.
    X_ca_full, units_dict_tmp = _process_aviary_metadata(X_ca, units_dict, aviary_input_names, aviary_meta_data)
    Y_ca_full, units_dict_full = _process_aviary_metadata(Y_ca, units_dict_tmp, aviary_output_names, aviary_meta_data)

    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(compute_adable, Y_ca_full, ad_backend, X_ca_full)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca_full,)), only(getaxes(X_ca_full))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca_full, ComplexF64)
    Y_ca_cs = similar(Y_ca_full, ComplexF64)

    return SparseADExplicitComp{true}(ad_backend, compute_adable, X_ca_full, Y_ca_full, J_ca_sparse, prep, rcdict, units_dict_full, tags_dict, X_ca_cs, Y_ca_cs, aviary_input_names, aviary_output_names)
end

function SparseADExplicitComp(ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), aviary_input_names=Dict{Symbol,String}(), aviary_output_names=Dict{Symbol,String}(), aviary_meta_data=Dict{String,Any}()) where {TAD<:ADTypes.AutoSparse}
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = let params=params
        (X,)->begin
            return f(X, params)
        end
    end

    # Process the Aviary metadata for the inputs.
    X_ca_full, units_dict_tmp = _process_aviary_metadata(X_ca, units_dict, aviary_input_names, aviary_meta_data)

    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(compute_adable, ad_backend, X_ca_full)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = compute_adable(X_ca_full)

    # Process the Aviary metadata for the outputs.
    Y_ca_full, units_dict_full = _process_aviary_metadata(Y_ca, units_dict_tmp, aviary_output_names, aviary_meta_data)

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca_full,)), only(getaxes(X_ca_full))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca_full, ComplexF64)

    return SparseADExplicitComp{false}(ad_backend, compute_adable, X_ca_full, J_ca_sparse, prep, rcdict, units_dict_full, tags_dict, X_ca_cs, aviary_input_names, aviary_output_names)
end

get_rows_cols_dict(comp::SparseADExplicitComp) = comp.rcdict

function get_partials_data(self::SparseADExplicitComp)
    rcdict = get_rows_cols_dict(self)
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]
        # Convert from 1-based to 0-based indexing.
        rows0based = rows .- 1
        cols0based = cols .- 1
        push!(partials_data, OpenMDAOCore.PartialsData(get_aviary_output_name(self, output_name), get_aviary_input_name(self, input_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end

_maybe_nonzeros(A::AbstractArray) = A
_maybe_nonzeros(A::AbstractSparseArray) = nonzeros(A)
_maybe_nonzeros(A::Base.ReshapedArray{T,N,P}) where {T,N,P<:AbstractSparseArray} = nonzeros(parent(A))

function OpenMDAOCore.compute_partials!(self::SparseADExplicitComp{true}, inputs, partials)
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
    J_ca_sparse = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    rcdict = get_rows_cols_dict(self)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca_sparse[oname, iname])
            
            # Need to reshape the subjacobian to correspond to the rows and cols.
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)

            # Grab the entry in partials we're interested in, and write the data we want to it.
            rows, cols = rcdict[oname, iname]

            # This gets the underlying Vector that stores the nonzero entries in the current sub-Jacobian that OpenMDAO sees.
            iname_aviary = get_aviary_input_name(self, iname)
            oname_aviary = get_aviary_output_name(self, oname)
            Jsub_out = partials[oname_aviary, iname_aviary]

            # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
            Jsub_out_vec = _maybe_nonzeros(Jsub_out)

            # Now write the non-zero entries to Jsub_out_vec.
            Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
        end
    end

    return nothing
end

function OpenMDAOCore.compute_partials!(self::SparseADExplicitComp{false}, inputs, partials)
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
    J_ca_sparse = get_jacobian_ca(self)
    prep = get_prep(self)
    ad_backend = get_backend(self)
    DifferentiationInterface.jacobian!(f, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    raxis, caxis = getaxes(J_ca_sparse)
    rcdict = get_rows_cols_dict(self)
    for oname in keys(raxis)
        for iname in keys(caxis)
            # Grab the subjacobian we're interested in.
            Jsub_in = @view(J_ca_sparse[oname, iname])
            
            # Need to reshape the subjacobian to correspond to the rows and cols.
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)

            # Grab the entry in partials we're interested in, and write the data we want to it.
            rows, cols = rcdict[oname, iname]

            # This gets the underlying Vector that stores the nonzero entries in the current sub-Jacobian that OpenMDAO sees.
            iname_aviary = get_aviary_input_name(self, iname)
            oname_aviary = get_aviary_output_name(self, oname)
            Jsub_out = partials[oname_aviary, iname_aviary]

            # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
            Jsub_out_vec = _maybe_nonzeros(Jsub_out)

            # Now write the non-zero entries to Jsub_out_vec.
            Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
        end
    end

    return nothing
end

has_setup_partials(self::SparseADExplicitComp) = false
has_compute_partials(self::SparseADExplicitComp) = true
has_compute_jacvec_product(self::SparseADExplicitComp) = false
