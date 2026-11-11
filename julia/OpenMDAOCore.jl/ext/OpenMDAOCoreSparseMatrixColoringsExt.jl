module OpenMDAOCoreSparseMatrixColoringsExt

using ComponentArrays: ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata
using ADTypes: ADTypes
using DifferentiationInterface: DifferentiationInterface
using SparseArrays: sparse, findnz, nonzeros, AbstractSparseArray
using SparseMatrixColorings: SparseMatrixColorings
using Random: rand!

using OpenMDAOCore: OpenMDAOCore

# Sparse-specific utilities --------------------------------------------------

"""
    _get_rows_cols_dict_from_sparsity(J::ComponentMatrix)

Get a `Dict` of the non-zero row and column indices for a sparsity pattern defined by a `ComponentMatrix` representation of a Jacobian.
"""
function OpenMDAOCore._get_rows_cols_dict_from_sparsity(J::ComponentMatrix)
    rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
    raxis, caxis = getaxes(J)
    for input_name in keys(caxis)
        for output_name in keys(raxis)
            # Grab the subjacobian we're interested in.
            Jsub = J[output_name, input_name]
            # We have to re-sparsify `Jsub` sometimes.
            # For example, if the output is a 2D array and the input is scalar, `Jsub` will be a reshaped sparse vector, which doesn't work with `findnz`.
            # Passing `Jsub` in that case to `sparse` converts it to a `SparseMatrixCSC`, which works with `findnz`.
            # Unfortunately that does appear to copy memory.
            # It'd be nice if I didn't have to do that.
            # But should be pretty small if the sub-jacobians are actually sparse.
            if typeof(Jsub) <: Number
                # Both input and output is scalar, so check if this scalar sub-Jacobian is zero or not.
                if Jsub ≈ zero(Jsub)
                    rows = cols = Vector{Int}()
                else
                    rows = cols = [1]
                end
            else
                Jsub_reshape = reshape(Jsub, length(raxis[output_name]), length(caxis[input_name]))
                rows, cols, vals = findnz(sparse(Jsub_reshape))
            end
            rcdict[output_name, input_name] = rows, cols
        end
    end

    return rcdict
end

function OpenMDAOCore.ca2strdict_sparse(ca::ComponentMatrix)
    T = eltype(ca)
    raxis, caxis = getaxes(ca)
    out = Dict{Tuple{String,String}, Vector{T}}()
    for input_name in keys(caxis)
        for output_name in keys(raxis)
            Jsub = ca[output_name, input_name]
            Jsub_reshape = reshape(Jsub, length(raxis[output_name]), length(caxis[input_name]))
            data_sparse = sparse(Jsub_reshape)
            out[string(output_name), string(input_name)] = nonzeros(data_sparse)
        end
    end
    return out
end

OpenMDAOCore._maybe_nonzeros(A::AbstractSparseArray) = nonzeros(A)
OpenMDAOCore._maybe_nonzeros(A::Base.ReshapedArray{T,N,P}) where {T,N,P<:AbstractSparseArray} = nonzeros(parent(A))

function OpenMDAOCore._assemble_sparse_partials!(partials, J_ca, blocks)
    J_data = OpenMDAOCore.getdata(J_ca)
    for block in blocks
        local output
        try
            output = partials[block.output_name, block.input_name]
        catch e
            isa(e, KeyError) || rethrow()
        else
            output = OpenMDAOCore._maybe_nonzeros(output)
            @inbounds for i in eachindex(block.rows, block.cols)
                output[i] = J_data[block.rows[i], block.cols[i]]
            end
        end
    end
    return nothing
end

# PerturbedDenseSparsityDetector --------------------------------------------
# The `PerturbedDenseSparsityDetector` *type* (struct, `show`, constructor) is
# declared in the parent `OpenMDAOCore` module so it can be imported without the
# extension loaded. Only the sparsity-detection *methods* below require
# `SparseArrays` and stay here.

## Direct

function ADTypes.jacobian_sparsity(f, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    rand!(perturb1)
    rand!(perturb2)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
    Jabs = abs.(DifferentiationInterface.jacobian(f, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
        Jabs .+= abs.(DifferentiationInterface.jacobian(f, backend, x_perturb))
    end

    return sparse(Jabs .> atol)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    rand!(perturb1)
    rand!(perturb2)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
    Jabs = abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
        Jabs .+= abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))
    end

    return sparse(Jabs .> atol)
end

function jacobian_sparsity!(Jabs, f!, y, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    rand!(perturb1)
    rand!(perturb2)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
    Jabs .= abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
        foo = abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))
        Jabs .+= foo
    end

    return nothing
end

function ADTypes.hessian_sparsity(f, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    rand!(perturb1)
    rand!(perturb2)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
    Habs = abs.(DifferentiationInterface.hessian(f, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb
        Habs .+= abs.(DifferentiationInterface.hessian(f, backend, x_perturb))
    end

    return sparse(Habs .> atol)
end

function ADTypes.jacobian_sparsity(f, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector
    y = f(x)

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    n, m = length(x), length(y)
    IJ = Vector{Tuple{Int,Int}}()

    # Need to make sure I don't add duplicates to I and J.
    # I guess the only way to do that is just to check.
    # It would be cool if I could skip adding non-zero entries for rows/columns that I've already identified as all non-sparse.
    for _ in 1:nevals
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb

        if DifferentiationInterface.pushforward_performance(backend) isa DifferentiationInterface.PushforwardFast
            p = similar(y)
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                DifferentiationInterface.pushforward!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                DifferentiationInterface.pullback!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(y, i),))
                for kj in LinearIndices(p)
                    if (abs(p[kj]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        end
    end

    I = getindex.(IJ, 1)
    J = getindex.(IJ, 2)
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)

    n, m = length(x), length(y)
    IJ = Vector{Tuple{Int,Int}}()

    for _ in 1:nevals
        rand!(perturb1)
        rand!(perturb2)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb

        if DifferentiationInterface.pushforward_performance(backend) isa DifferentiationInterface.PushforwardFast
            p = similar(y)
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                DifferentiationInterface.pushforward!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                DifferentiationInterface.pullback!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(y, i),))
                for kj in LinearIndices(p)
                    if (abs(p[kj]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        end
    end

    I = getindex.(IJ, 1)
    J = getindex.(IJ, 2)
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.hessian_sparsity(f, x, detector::OpenMDAOCore.PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector

    x_perturb = similar(x)
    perturb1 = similar(x)
    perturb2 = similar(x)
    p = similar(x)

    n = length(x)
    IJ = Vector{Tuple{Int,Int}}()
    for _ in 1:nevals
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb1 .- 0.5)).*x .+ (perturb2 .- 0.5).*abs_x_perturb

        prep = DifferentiationInterface.prepare_hvp_same_point(f, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),))
        for (kj, j) in enumerate(eachindex(x_perturb))
            DifferentiationInterface.hvp!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, j),))
            for ki in LinearIndices(p)
                if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                    push!(IJ, (ki, kj))
                end
            end
        end
    end

    I = getindex.(IJ, 1)
    J = getindex.(IJ, 2)
    return sparse(I, J, ones(Bool, length(I)), n, n)
end

# ADExplicitComp{SparseFlavor} ------------------------------------------------

# The `ADExplicitComp{SparseFlavor, ...}` *type* and the shared accessors
# (`get_jacobian_ca`, `get_rows_cols_dict`, `has_*`) are declared in the parent
# `OpenMDAOCore` module (in `abstract_ad.jl`) so they can be imported without
# the extension loaded. Only the `SparseFlavor` constructors and the
# `compute_partials!`/`setup_partials`/`_update_prep`/`get_partials_data` methods
# below require `SparseArrays`/`SparseMatrixColorings` and stay here.

"""
    ADExplicitComp(::SparseFlavor, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an in-place [`SparseFlavor`](@ref) [`ADExplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
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
function OpenMDAOCore.ADExplicitComp(::OpenMDAOCore.SparseFlavor, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AutoSparse}

    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable = OpenMDAOCore._make_compute_adable(Val(true), f!, params)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)
    else
        # No point in getting a "good" prep when we don't know all the shapes.
        J_ca_sparse = nothing
        prep = nothing
        rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
        deriv_prep = OpenMDAOCore.SparseDerivPrep(J_ca_sparse, prep, rcdict)
        X_ca_cs = Y_ca_cs = nothing
    end

    return OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, true}(ad_backend, f!, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, OpenMDAOCore.InputCopyPrep(X_ca), OpenMDAOCore.InputCopyPrep(Y_ca))
end

"""
    ADExplicitComp(::SparseFlavor, ad_backend, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false)

Create an out-of-place [`SparseFlavor`](@ref) [`ADExplicitComp`](@ref).

# Positional Arguments
* `ad_backend`: `<:ADTypes.AutoSparse` automatic differentation "backend" library
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
function OpenMDAOCore.ADExplicitComp(::OpenMDAOCore.SparseFlavor, ad_backend::TAD, f, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false) where {TAD<:ADTypes.AutoSparse}

    compute_adable = OpenMDAOCore._make_compute_adable(Val(false), f, params)

    Y_ca = compute_adable(X_ca)

    # Get the prep-related stuff.
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, X_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, X_ca)
    else
        J_ca_sparse = nothing
        prep = nothing
        rcdict = Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}()
        deriv_prep = OpenMDAOCore.SparseDerivPrep(J_ca_sparse, prep, rcdict)
        X_ca_cs = nothing
    end

    Y_ca_cs = nothing
    return OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, false}(ad_backend, f, params, compute_adable, X_ca, Y_ca, deriv_prep,
        units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, X_ca_cs, Y_ca_cs, OpenMDAOCore.InputCopyPrep(X_ca), OpenMDAOCore.InputCopyPrep(Y_ca))
end

function _get_sparse_prep_stuff(ad_backend, f!, Y_ca, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f!, Y_ca, ad_backend, X_ca)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = OpenMDAOCore._get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return OpenMDAOCore.SparseDerivPrep(J_ca_sparse, prep, rcdict), X_ca_cs, Y_ca_cs
end

function _get_sparse_prep_stuff(ad_backend, f, X_ca)
    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(f, ad_backend, X_ca)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Need the output component vector to define the axes of the Jacobian.
    Y_ca = f(X_ca)

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = OpenMDAOCore._get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca_full and Y_ca_full arrays.
    X_ca_cs = similar(X_ca, ComplexF64)

    return OpenMDAOCore.SparseDerivPrep(J_ca_sparse, prep, rcdict), X_ca_cs
end

function OpenMDAOCore._update_prep(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        X_ca_old = OpenMDAOCore.get_input_ca(self)
        Y_ca_old = OpenMDAOCore.get_output_ca(self)

        # Create a new versions of `X_ca_old` that have the correct sizes and default values.
        X_ca = OpenMDAOCore._resize_component_vector(X_ca_old, input_sizes)
        Y_ca = OpenMDAOCore._resize_component_vector(Y_ca_old, output_sizes)

        # Get the new sparsity stuff.
        ad_backend = OpenMDAOCore.get_backend(self)
        compute_adable = OpenMDAOCore._make_compute_adable(Val(true), self.func, self.params)
        deriv_prep, X_ca_cs, Y_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, Y_ca, X_ca)

        self = OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, true}(ad_backend, self.func, self.params, compute_adable, X_ca, Y_ca, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, Y_ca_cs, OpenMDAOCore.InputCopyPrep(X_ca), OpenMDAOCore.InputCopyPrep(Y_ca))
    end

    return self
end

function OpenMDAOCore._update_prep(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})

    if length(input_sizes) > 0
        X_ca_old = OpenMDAOCore.get_input_ca(self)

        X_ca = OpenMDAOCore._resize_component_vector(X_ca_old, input_sizes)

        # Get the new sparsity stuff.
        ad_backend = OpenMDAOCore.get_backend(self)
        compute_adable = OpenMDAOCore._make_compute_adable(Val(false), self.func, self.params)
        deriv_prep, X_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, X_ca)

        self = OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, false}(ad_backend, self.func, self.params, compute_adable, X_ca, nothing, deriv_prep,
            self.units_dict, self.tags_dict, self.shape_by_conn_dict, self.copy_shape_dict, X_ca_cs, nothing, OpenMDAOCore.InputCopyPrep(X_ca), OpenMDAOCore.InputCopyPrep(compute_adable(X_ca)))
    end

    return self
end

function  _get_py_indices_non_flat(shape)
    # First, get the flattened 0-based indices.
    idx_flat = 0:(prod(shape)-1)

    # Now reshape it into the reversed dimenions, then permute the dimensions.
    # This will give us an array that has the shape indicated by the `shape` argument to this function, but filled with the appropriated indices for a zero-based, Python-ordered (aka row-major ordered) array.
    return PermutedDimsArray(reshape(idx_flat, reverse(shape)), length(shape):-1:1)
end

function  _get_py_indices(shape)
    idx_non_flat = _get_py_indices_non_flat(shape)
    # Now create a flattened view:
    return view(idx_non_flat, :)
end

function OpenMDAOCore.get_partials_data(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor})
    rcdict = OpenMDAOCore.get_rows_cols_dict(self)
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    X_ca = OpenMDAOCore.get_input_ca(self)
    Y_ca = OpenMDAOCore.get_output_ca(self)
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]

        # Create an array that has the same shape as the input or output but with Python flat indices™ as values.
        input_idx_py = _get_py_indices(size(X_ca[input_name]))
        output_idx_py = _get_py_indices(size(Y_ca[output_name]))

        # Translate the Julia-ordered, 1-based rows and cols to Python-ordered, 0-based rows and cols.
        cols0based = getindex.(Ref(input_idx_py), cols)
        rows0based = getindex.(Ref(output_idx_py), rows)

        push!(partials_data, OpenMDAOCore.PartialsData(string(output_name), string(input_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end

function OpenMDAOCore.setup_partials(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor}, input_sizes, output_sizes)

    input_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in input_sizes)
    output_sizes_ca = Dict{Symbol,Any}(Symbol(k)=>sz for (k, sz) in output_sizes)

    self_new = OpenMDAOCore._update_prep(self, input_sizes_ca, output_sizes_ca)

    # Now finally get the partials data.
    return self_new, OpenMDAOCore.get_partials_data(self_new)
end

function OpenMDAOCore.compute_partials!(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, true}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = OpenMDAOCore.get_input_ca(self)
    OpenMDAOCore._copy_component_inputs!(OpenMDAOCore.getdata(X_ca), inputs, self.input_copy_prep)

    # Get the Jacobian.
    f! = OpenMDAOCore.get_callback(self)
    Y_ca = OpenMDAOCore.get_output_ca(self)
    J_ca_sparse = OpenMDAOCore.get_jacobian_ca(self)
    prep = OpenMDAOCore.get_prep(self)
    ad_backend = OpenMDAOCore.get_backend(self)
    prep === nothing && throw(_prep_not_created_error(self, "compute_partials!"))
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    OpenMDAOCore._assemble_sparse_partials!(partials, J_ca_sparse, self.deriv_prep.blocks)

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OpenMDAOCore.ADExplicitComp{OpenMDAOCore.SparseFlavor, false}, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = OpenMDAOCore.get_input_ca(self)
    OpenMDAOCore._copy_component_inputs!(OpenMDAOCore.getdata(X_ca), inputs, self.input_copy_prep)

    # Get the Jacobian.
    f = OpenMDAOCore.get_callback(self)
    J_ca_sparse = OpenMDAOCore.get_jacobian_ca(self)
    prep = OpenMDAOCore.get_prep(self)
    ad_backend = OpenMDAOCore.get_backend(self)
    prep === nothing && throw(_prep_not_created_error(self, "compute_partials!"))
    DifferentiationInterface.jacobian!(f, J_ca_sparse, prep, ad_backend, X_ca)

    # Extract the derivatives from `J_ca_sparse` and put them in `partials`.
    OpenMDAOCore._assemble_sparse_partials!(partials, J_ca_sparse, self.deriv_prep.blocks)

    return nothing
end

# ---------------------------------------------------------------------------
# ADImplicitComp{SparseFlavor} — implicit sparse AD components.
#
# As with the explicit sparse components, the `ADImplicitComp{SparseFlavor}`
# *type* and the `has_*` methods are declared in the main package, and the
# constructors and the `linearize!` methods are provided here.
# ---------------------------------------------------------------------------

OpenMDAOCore.get_jacobian_ca(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor}) = comp.deriv_prep.J_ca_sparse
OpenMDAOCore.get_rows_cols_dict(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor}) = comp.deriv_prep.rcdict

"""
    ADImplicitComp(::SparseFlavor, ::Val{true}, ad_backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an in-place [`SparseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.jacobian!` into a sparse Jacobian.
"""
function OpenMDAOCore.ADImplicitComp(::OpenMDAOCore.SparseFlavor, ::Val{true}, ad_backend::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AutoSparse}

    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = OpenMDAOCore._make_implicit_compute_adable(true, f!, params, Y_range, X_range, Y_axes, X_axes)
    R_ca = similar(Y_ca)

    # The explicit sparse prep builder works for implicit components, too:
    # the implicit `compute_adable(R, YX)` closure already has
    # DifferentiationInterface's in-place (`f!(y, x)`) form, so we pass `R_ca`
    # as the "output" argument (`Y_ca`) and the combined `YX_ca` as the
    # "input" argument (`X_ca`).
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, R_ca, YX_ca)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup).
        deriv_prep = OpenMDAOCore.SparseDerivPrep(nothing, nothing, Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}())
        YX_ca_cs = R_ca_cs = nothing
    end

    return OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, true}(ad_backend, compute_adable, f!, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, OpenMDAOCore.InputCopyPrep(Y_ca), OpenMDAOCore.InputCopyPrep(X_ca), R_ca === nothing ? OpenMDAOCore.InputCopyPrep(Y_ca) : OpenMDAOCore.InputCopyPrep(R_ca); guess_func=guess_func)
end

"""
    ADImplicitComp(::SparseFlavor, ::Val{false}, ad_backend, f, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=..., tags_dict=..., shape_by_conn_dict=..., copy_shape_dict=..., force_skip_prep=false)

Create an out-of-place [`SparseFlavor`](@ref) [`ADImplicitComp`](@ref).
Derivatives are computed via `DifferentiationInterface.jacobian!` into a sparse Jacobian.
"""
function OpenMDAOCore.ADImplicitComp(::OpenMDAOCore.SparseFlavor, ::Val{false}, ad_backend::TAD, f, Y_ca::ComponentVector, X_ca::ComponentVector;
        params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), shape_by_conn_dict=Dict{Symbol,Bool}(), copy_shape_dict=Dict{Symbol,Symbol}(), force_skip_prep=false, guess_func=nothing) where {TAD<:ADTypes.AutoSparse}

    common_keys = intersect(keys(Y_ca), keys(X_ca))
    if !isempty(common_keys)
        throw(ArgumentError("State and input ComponentVectors share the following key(s): $(collect(common_keys)). State and input names must be distinct."))
    end

    YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
    Y_range = 1:length(Y_ca)
    X_range = length(Y_ca)+1:length(YX_ca)
    Y_axes = getaxes(Y_ca)
    X_axes = getaxes(X_ca)

    compute_adable = OpenMDAOCore._make_implicit_compute_adable(false, f, params, Y_range, X_range, Y_axes, X_axes)
    if (!any(values(shape_by_conn_dict))) && (length(copy_shape_dict) == 0) && (!force_skip_prep)
        deriv_prep, YX_ca_cs = _get_sparse_prep_stuff(ad_backend, compute_adable, YX_ca)
    else
        # Shapes not yet known: defer to `setup_partials` (called by OpenMDAO
        # during problem setup).
        deriv_prep = OpenMDAOCore.SparseDerivPrep(nothing, nothing, Dict{Tuple{Symbol,Symbol}, Tuple{Vector{Int},Vector{Int}}}())
        YX_ca_cs = nothing
    end

    R_ca = nothing
    R_ca_cs = nothing

    return OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, false}(ad_backend, compute_adable, f, params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep, units_dict, tags_dict, shape_by_conn_dict, copy_shape_dict, Y_range, X_range, Y_axes, X_axes, OpenMDAOCore.InputCopyPrep(Y_ca), OpenMDAOCore.InputCopyPrep(X_ca), R_ca === nothing ? OpenMDAOCore.InputCopyPrep(Y_ca) : OpenMDAOCore.InputCopyPrep(R_ca); guess_func=guess_func)
end

# Compute dR/d(Y, X) with a single sparse `jacobian!` call, then scatter the
# sub-Jacobians to the `partials` dict using the sparsity pattern.
function OpenMDAOCore._update_prep(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, true}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = OpenMDAOCore.get_output_ca(comp)
        X_ca_old = OpenMDAOCore.get_input_ca(comp)

        Y_ca = OpenMDAOCore._resize_component_vector(Y_ca_old, output_sizes)
        X_ca = OpenMDAOCore._resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        R_ca = similar(Y_ca)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = OpenMDAOCore._make_implicit_compute_adable(true, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        deriv_prep, YX_ca_cs, R_ca_cs = _get_sparse_prep_stuff(OpenMDAOCore.get_backend(comp), compute_adable, R_ca, YX_ca)

        comp = OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, true}(OpenMDAOCore.get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, OpenMDAOCore.InputCopyPrep(Y_ca), OpenMDAOCore.InputCopyPrep(X_ca), R_ca === nothing ? OpenMDAOCore.InputCopyPrep(Y_ca) : OpenMDAOCore.InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function OpenMDAOCore.linearize!(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, true}, inputs, outputs, partials)
    YX_ca = OpenMDAOCore.get_combined_ca(comp)
    OpenMDAOCore._copy_component_inputs!(view(OpenMDAOCore.getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    OpenMDAOCore._copy_component_inputs!(view(OpenMDAOCore.getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    f! = OpenMDAOCore.get_callback(comp)
    R_ca = OpenMDAOCore.get_residual_ca(comp)
    J_ca = OpenMDAOCore.get_jacobian_ca(comp)
    prep = OpenMDAOCore.get_prep(comp)
    ad_backend = OpenMDAOCore.get_backend(comp)
    J_ca === nothing && throw(_prep_not_created_error(comp, "linearize!"))
    DifferentiationInterface.jacobian!(f!, R_ca, J_ca, prep, ad_backend, YX_ca)

    OpenMDAOCore._assemble_sparse_partials!(partials, J_ca, comp.deriv_prep.blocks)

    return nothing
end

function OpenMDAOCore._update_prep(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, false}, input_sizes::AbstractDict{Symbol,<:Any}, output_sizes::AbstractDict{Symbol,<:Any})
    if (length(input_sizes) > 0) || (length(output_sizes) > 0)
        Y_ca_old = OpenMDAOCore.get_output_ca(comp)
        X_ca_old = OpenMDAOCore.get_input_ca(comp)

        Y_ca = OpenMDAOCore._resize_component_vector(Y_ca_old, output_sizes)
        X_ca = OpenMDAOCore._resize_component_vector(X_ca_old, input_sizes)

        YX_ca = ComponentVector(; (k => Y_ca[k] for k in keys(Y_ca))..., (k => X_ca[k] for k in keys(X_ca))...)
        Y_range = 1:length(Y_ca)
        X_range = length(Y_ca)+1:length(YX_ca)
        Y_axes = getaxes(Y_ca)
        X_axes = getaxes(X_ca)

        compute_adable = OpenMDAOCore._make_implicit_compute_adable(false, comp.func, comp.params, Y_range, X_range, Y_axes, X_axes)
        deriv_prep, YX_ca_cs = _get_sparse_prep_stuff(OpenMDAOCore.get_backend(comp), compute_adable, YX_ca)

        R_ca = nothing
        R_ca_cs = nothing

        comp = OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, false}(OpenMDAOCore.get_backend(comp), compute_adable, comp.func, comp.params, YX_ca, R_ca, YX_ca_cs, R_ca_cs, deriv_prep,
            comp.units_dict, comp.tags_dict, comp.shape_by_conn_dict, comp.copy_shape_dict, Y_range, X_range, Y_axes, X_axes, OpenMDAOCore.InputCopyPrep(Y_ca), OpenMDAOCore.InputCopyPrep(X_ca), R_ca === nothing ? OpenMDAOCore.InputCopyPrep(Y_ca) : OpenMDAOCore.InputCopyPrep(R_ca); guess_func=comp.guess_func)
    end

    return comp
end

function OpenMDAOCore.linearize!(comp::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor, false}, inputs, outputs, partials)
    YX_ca = OpenMDAOCore.get_combined_ca(comp)
    OpenMDAOCore._copy_component_inputs!(view(OpenMDAOCore.getdata(YX_ca), comp.Y_range), outputs, comp.state_copy_prep)
    OpenMDAOCore._copy_component_inputs!(view(OpenMDAOCore.getdata(YX_ca), comp.X_range), inputs, comp.input_copy_prep)

    f = OpenMDAOCore.get_callback(comp)
    J_ca = OpenMDAOCore.get_jacobian_ca(comp)
    prep = OpenMDAOCore.get_prep(comp)
    ad_backend = OpenMDAOCore.get_backend(comp)
    J_ca === nothing && throw(_prep_not_created_error(comp, "linearize!"))
    DifferentiationInterface.jacobian!(f, J_ca, prep, ad_backend, YX_ca)

    OpenMDAOCore._assemble_sparse_partials!(partials, J_ca, comp.deriv_prep.blocks)

    return nothing
end

# `get_partials_data` for implicit sparse components: one `PartialsData` entry
# per (residual, wrt) pair in the sparsity pattern, with rows/cols translated
# from Julia 1-based to Python 0-based indices (as the explicit version does).
function OpenMDAOCore.get_partials_data(self::OpenMDAOCore.ADImplicitComp{OpenMDAOCore.SparseFlavor})
    rcdict = OpenMDAOCore.get_rows_cols_dict(self)
    partials_data = Vector{OpenMDAOCore.PartialsData}()
    Y_ca = OpenMDAOCore.get_output_ca(self)
    X_ca = OpenMDAOCore.get_input_ca(self)
    y_keys = Set(Symbol.(keys(Y_ca)))
    for (residual_name, wrt_name) in keys(rcdict)
        rows, cols = rcdict[residual_name, wrt_name]

        # The `of` side is always a residual, which has the same structure as
        # the corresponding state/output variable.
        output_idx_py = _get_py_indices(size(Y_ca[residual_name]))
        rows0based = getindex.(Ref(output_idx_py), rows)

        # The `wrt` side can be either a state (Y) or an input (X) variable.
        if wrt_name in y_keys
            wrt_idx_py = _get_py_indices(size(Y_ca[wrt_name]))
        else
            wrt_idx_py = _get_py_indices(size(X_ca[wrt_name]))
        end
        cols0based = getindex.(Ref(wrt_idx_py), cols)

        push!(partials_data, OpenMDAOCore.PartialsData(string(residual_name), string(wrt_name); rows=rows0based, cols=cols0based))
    end

    return partials_data
end


end # module
