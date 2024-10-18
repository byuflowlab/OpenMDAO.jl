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
get_units(comp::AbstractAutoSparseForwardDiffExplicitComp, varname) = get(comp.units_dict, varname, nothing)
get_tags(comp::AbstractAutoSparseForwardDiffExplicitComp, varname) = get(comp.tags_dict, varname, nothing)
get_rows_cols_dict(comp::AbstractAutoSparseForwardDiffExplicitComp) = comp.rcdict

function get_input_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::AbstractAutoSparseForwardDiffExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
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
        # This requires that each output is at least a vector.
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
            Jsub_in = @view(J_ca_sparse[oname, iname])
            
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

"""
    SimpleAutoSparseForwardDiffExplicitComp{TCompute,TX,TY,TJ,TCache,TRCDict,TXCS,TYCS} <: AbstractAutoSparseForwardDiffExplicitComp

An `<:AbstractAutoSparseForwardDiffExplicitComp` with a simplified interface.

# Fields
* `compute_forwarddiffable!::TCompute`: function of the form `compute_forwarddiffable!(Y, X)` compatible with ForwardDiff.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::TX`: `ComponentVector` of inputs
* `Y_ca::TY`: `ComponentVector` of outputs
* `J_ca_sparse::TJ`: Sparse `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `jac_cache::TCache`: Cache object used by `SparseDiffTools`.
* `rcdict::TRCDict`: `Dict{Tuple{Symbol,Sympol}, Tuple{Vector{Int}, Vector{Int}}` mapping sub-Jacobians of the form `(:output_name, :input_name)` to `Vector`s of non-zero row and column indices
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `X_ca::TXCS`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::TXCS`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct SimpleAutoSparseForwardDiffExplicitComp{TCompute,TX,TY,TJ,TCache,TRCDict,TXCS,TYCS} <: AbstractAutoSparseForwardDiffExplicitComp
    compute_forwarddiffable!::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    jac_cache::TCache
    rcdict::TRCDict
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS
end

"""
    SimpleAutoSparseForwardDiffExplicitComp(f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), num_jac_perturbs=3, rel_jac_perturb=0.001)

Create a `SimpleAutoSparseForwardDiffExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `J_ca`: `ComponentMatrix` defining the sparsity of of the Jacobian of `Y_ca` with respect to `X_ca`. If `nothing`, the sparsity will be computed using `generate_perturbed_jacobian!`.
* `num_jac_perturbs`: number of times the Jacobian will be evaluated using perturbed inputs to detect the sparsity of the Jacobian
* `rel_jac_perturb`: relative amount of random perturbation of the inputs during the Jacobian sparsity detection.
"""
function SimpleAutoSparseForwardDiffExplicitComp(f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), J_ca=nothing, num_jac_perturbs=3, rel_jac_perturb=0.001)
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_forwarddiffable! = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    if J_ca === nothing
        # Create a dense ComponentMatrix from the input and output arrays.
        _J_ca = Y_ca.*X_ca'

        # Hopefully perturb the Jacobian a few times to detect the sparsity.
        generate_perturbed_jacobian!(_J_ca, compute_forwarddiffable!, Y_ca, X_ca, num_jac_perturbs, rel_jac_perturb)
    else
        _J_ca = J_ca
    end

    # Create a sparse matrix version of _J_ca.
    J_ca_sparse = ComponentMatrix(sparse(getdata(_J_ca)), getaxes(_J_ca))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Get some colors!
    colors = matrix_colors(getdata(J_ca_sparse))

    # Create the cache object for `forwarddiff_color_jacobian!`.
    jac_cache = ForwardColorJacCache(f!, X_ca; dx=Y_ca, colorvec=colors, sparsity=getdata(J_ca_sparse))

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return SimpleAutoSparseForwardDiffExplicitComp(compute_forwarddiffable!, X_ca, Y_ca, J_ca_sparse, jac_cache, rcdict, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
end

"""
    PerturbedDenseSparsityDetector

Tweaked version of [`DenseSparsityDetector`](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/api/#DifferentiationInterface.DenseSparsityDetector) sparsity pattern detector satisfying the [detection API](https://sciml.github.io/ADTypes.jl/stable/#Sparse-AD) of [ADTypes.jl](https://github.com/SciML/ADTypes.jl) that evaluates the Jacobian multiple times using a perturbed input vector.
Specifically, input vector `x` will be perturbed via

```julia
    x_perturb = (1 .+ rel_x_perturb.*perturb).*x
```

where `perturb` is a random `Vector` of numbers ranging from `-0.5` to `0.5`, and `rel_x_perturb` is a relative perturbation magnitude specified by the user.

All of the caveats associated with the performance of `DenseSparsityDetector` apply to `PerturbedDenseSparsityDetector`, since it essentially does the same thing as `DenseSparsityDetector` multiple times.
The nonzeros in a Jacobian or Hessian are detected by computing the relevant matrix with _dense_ AD, and thresholding the entries with a given tolerance (which can be numerically inaccurate).
This process can be very slow, and should only be used if its output can be exploited multiple times to compute many sparse matrices.

!!! danger
    In general, the sparsity pattern you obtain can depend on the provided input `x`. If you want to reuse the pattern, make sure that it is input-agnostic.
    Perturbing the input vector should hopefully guard against getting "unlucky" and finding zero Jacobian entries that aren't actually zero for all `x`, but is of course problem-dependent.

# Fields

- `backend::AbstractADType` is the dense AD backend used under the hood
- `atol::Float64` is the minimum magnitude of a matrix entry to be considered nonzero
- `nevals::Int=3` is the number of times the Jacobian will be evaluated using the perturbed input `x`
- `rel_x_perturb=0.001`: is the relative magnitude of the `x` perturbation.

# Constructor

    PerturbedDenseSparsityDetector(backend; atol, method=:iterative, nevals=3, rel_x_perturb=0.001)

The keyword argument `method::Symbol` can be either:

- `:iterative`: compute the matrix in a sequence of matrix-vector products (memory-efficient)
- `:direct`: compute the matrix all at once (memory-hungry but sometimes faster).

Note that the constructor is type-unstable because `method` ends up being a type parameter of the `PerturbedDenseSparsityDetector` object (this is not part of the API and might change).

"""
struct PerturbedDenseSparsityDetector{method,B,TRelXPerturb} <: ADTypes.AbstractSparsityDetector
    backend::B
    atol::Float64
    nevals::Int
    rel_x_perturb::TRelXPerturb
end

function Base.show(io::IO, detector::PerturbedDenseSparsityDetector{method}) where {method}
    (; backend, atol, nevals, rel_x_perturb) = detector
    return print(
        io,
        PerturbedDenseSparsityDetector,
        "(",
        repr(backend; context=io),
        "; atol=$atol, method=",
        repr(method; context=io),
        "nevals=$nevals, rel_x_perturb=$rel_x_perturb",
        ")",
    )
end

function PerturbedDenseSparsityDetector(
    backend::ADTypes.AbstractADType; atol::Float64, method::Symbol=:iterative, nevals=3, rel_x_perturb=0.001
)
    if !(method in (:iterative, :direct))
        throw(
            ArgumentError("The keyword `method` must be either `:iterative` or `:direct`.")
        )
    end

    if nevals < 1
        throw(
            ArgumentError("The keyword `nevals` should be > 0")
        )
    end

    return PerturbedDenseSparsityDetector{method,typeof(backend),typeof(rel_x_perturb)}(backend, atol, nevals, rel_x_perturb)
end

## Direct

function ADTypes.jacobian_sparsity(f, x, detector::PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb) = detector

    x_perturb = similar(x)
    perturb = similar(x)

    rand!(perturb)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
    Jabs = abs.(DifferentiationInterface.jacobian(f, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
        Jabs .+= abs.(DifferentiationInterface.jacobian(f, backend, x_perturb))
    end

    return sparse(Jabs .> atol)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb) = detector

    x_perturb = similar(x)
    perturb = similar(x)

    rand!(perturb)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
    Jabs = abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
        Jabs .+= abs.(DifferentiationInterface.jacobian(f!, y, backend, x_perturb))
    end

    return sparse(Jabs .> atol)
end

function ADTypes.hessian_sparsity(f, x, detector::PerturbedDenseSparsityDetector{:direct})
    (; backend, atol, nevals, rel_x_perturb) = detector

    x_perturb = similar(x)
    perturb = similar(x)

    rand!(perturb)
    x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
    Habs = abs.(DifferentiationInterface.hessian(f, backend, x_perturb))

    for i in 1:nevals-1
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x
        Habs .+= abs.(DifferentiationInterface.hessian(f, backend, x_perturb))
    end

    return sparse(Habs .> atol)
end

function ADTypes.jacobian_sparsity(f, x, detector::PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb) = detector
    y = f(x)

    x_perturb = similar(x)
    perturb = similar(x)

    n, m = length(x), length(y)
    IJ = Vector{Tuple{Int,Int}}()

    # Need to make sure I don't add duplicates to I and J.
    # I guess the only way to do that is just to check.
    # It would be cool if I could skip adding non-zero entries for rows/columns that I've already identified as all non-sparse.
    for _ in 1:nevals
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x

        if DifferentiationInterface.pushforward_performance(backend) isa DifferentiationInterface.PushforwardFast
            p = similar(y)
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                DifferentiationInterface.pushforward!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(backend, y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                DifferentiationInterface.pullback!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, y, i),))
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

function ADTypes.jacobian_sparsity(f!, y, x, detector::PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb) = detector

    x_perturb = similar(x)
    perturb = similar(x)

    n, m = length(x), length(y)
    IJ = Vector{Tuple{Int,Int}}()

    for _ in 1:nevals
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x

        if DifferentiationInterface.pushforward_performance(backend) isa DifferentiationInterface.PushforwardFast
            p = similar(y)
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                DifferentiationInterface.pushforward!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(backend, y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                DifferentiationInterface.pullback!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, y, i),))
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

function ADTypes.hessian_sparsity(f, x, detector::PerturbedDenseSparsityDetector{:iterative})
    (; backend, atol, nevals, rel_x_perturb) = detector

    x_perturb = similar(x)
    perturb = similar(x)
    p = similar(x)

    n = length(x)
    IJ = Vector{Tuple{Int,Int}}()
    for _ in 1:nevals
        rand!(perturb)
        x_perturb .= (1 .+ rel_x_perturb.*(perturb .- 0.5)).*x

        prep = DifferentiationInterface.prepare_hvp_same_point(f, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),))
        for (kj, j) in enumerate(eachindex(x_perturb))
            hvp!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
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

"""
    ADSparseExplicitComp{TAD,TCompute,TX,TY,TJ,TPrep,TRCDict,TXCS,TYCS} <: AbstractExplicitComp

An `<:AbstractAutoSparseForwardDiffExplicitComp` with a simplified interface.

# Fields
* `compute_adable!::TCompute`: function of the form `compute_adable!(Y, X)` compatible with DifferentiationInterface.jl that performs the desired computation, where `Y` and `X` are `ComponentVector`s of outputs and inputs, respectively
* `X_ca::TX`: `ComponentVector` of inputs
* `Y_ca::TY`: `ComponentVector` of outputs
* `J_ca_sparse::TJ`: Sparse `ComponentMatrix` of the Jacobian of `Y_ca` with respect to `X_ca`
* `prep::TPrep`: `DifferentiationInterface.jl` "preparation" object
* `rcdict::TRCDict`: `Dict{Tuple{Symbol,Sympol}, Tuple{Vector{Int}, Vector{Int}}` mapping sub-Jacobians of the form `(:output_name, :input_name)` to `Vector`s of non-zero row and column indices
* `units_dict::Dict{Symbol,String}`: mapping of variable names to units. Can be an empty `Dict` if units are not desired.
* `tags_dict::Dict{Symbol,Vector{String}`: mapping of variable names to `Vector`s of `String`s specifing variable tags.
* `X_ca::TXCS`: `ComplexF64` version of `X_ca` (for the complex-step method)
* `Y_ca::TXCS`: `ComplexF64` version of `Y_ca` (for the complex-step method)
"""
struct ADSparseExplicitComp{TAD,TCompute,TX,TY,TJ,TPrep,TRCDict,TXCS,TYCS} <: AbstractExplicitComp
    ad::TAD
    compute_adable!::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    prep::TPrep
    rcdict::TRCDict
    units_dict::Dict{Symbol,String}
    tags_dict::Dict{Symbol,Vector{String}}
    X_ca_cs::TXCS
    Y_ca_cs::TYCS
end

"""
    ADSparseExplicitComp(backend, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), num_jac_perturbs=3, rel_jac_perturb=0.001)

Create a `ADSparseExplicitComp` from a user-defined function and output and input `ComponentVector`s.

# Positional Arguments
* `f!`: function of the form `f!(Y_ca, X_ca, params)` which writes outputs to `Y_ca` using inputs `X_ca` and, optionally, parameters `params`.
* `Y_ca`: `ComponentVector` of outputs
* `X_ca`: `ComponentVector` of inputs

# Keyword Arguments
* `params`: parameters passed to the third argument to `f!`. Could be anything, or `nothing`, but the derivatives of `Y_ca` with respect to `params` will not be calculated
* `units_dict`: `Dict` mapping variable names (as `Symbol`s) to OpenMDAO units (expressed as `String`s)
* `tags_dict`: `Dict` mapping variable names (as `Symbol`s) to `Vector`s of OpenMDAO tags
* `J_ca`: `ComponentMatrix` defining the sparsity of of the Jacobian of `Y_ca` with respect to `X_ca`. If `nothing`, the sparsity will be computed using `generate_perturbed_jacobian!`.
* `num_jac_perturbs`: number of times the Jacobian will be evaluated using perturbed inputs to detect the sparsity of the Jacobian
* `rel_jac_perturb`: relative amount of random perturbation of the inputs during the Jacobian sparsity detection.
"""
function ADSparseExplicitComp(ad::TAD, f!, Y_ca::ComponentVector, X_ca::ComponentVector; params=nothing, units_dict=Dict{Symbol,String}(), tags_dict=Dict{Symbol,Vector{String}}(), J_ca=nothing, num_jac_perturbs=3, rel_jac_perturb=0.001) where {TAD<:ADTypes.AutoSparse}
    # Create a new user-defined function that captures the `params` argument.
    # https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    compute_adable! = let params=params
        (Y, X)->begin
            f!(Y, X, params)
            return nothing
        end
    end

    # So, some options here. 
    # If the user passes a sparse Jacobian, then we use that.
    # If they don't, then what we do should depend on the sparsity detector algorithm that they choose.
    # Ah, actually I bet the sparsity detection happens when the prep is created anyways...
    # I need to figure that out.
    # Yeah, so I need to be able to create the sparse jacobian.
    # Looks like that's possible.
    # Oh, hmm... actually there is a KnownSparsityDectory thingy.
    # Great!
    # So, what do I do about `generate_perturbed_jacobian!`?
    # I think the right thing to do would be to create a `<:ADTypes.AbstractSprsityDetector`.
    # It looks like the `DifferentiationInterface.DenseSparsityDetector` just evaluates the Jacobian at one `x`.
    # So maybe this approach would be the `PerturbedDenseSparsityDetector`.
    # It looks pretty easy to implement, but... hmm...

	# function ADTypes.jacobian_sparsity(f, x, detector::DenseSparsityDetector{:direct})
	# 	(; backend, atol) = detector
	# 	J = jacobian(f, backend, x)
	# 	return sparse(abs.(J) .> atol)
	# end
    #
    # function ADTypes.jacobian_sparsity(f!, y, x, detector::DenseSparsityDetector{:direct})
    #   (; backend, atol) = detector
    #   J = jacobian(f!, y, backend, x)
    #   return sparse(abs.(J) .> atol)
    # end

    # OK, that doesn't look too bad.
    # But who defines `jacobian`?
    # That's from DifferentiationInterface.jl, but that's OK, I'm going to use it anyway.
    # I just need the backend I guess.
    # Right.
    # OK, have that implemented now.
    # Now that everything is in the <:AutoSparse AD

    # if J_ca === nothing
    #     # Create a dense ComponentMatrix from the input and output arrays.
    #     _J_ca = Y_ca.*X_ca'

    #     # Hopefully perturb the Jacobian a few times to detect the sparsity.
    #     generate_perturbed_jacobian!(_J_ca, compute_adable!, Y_ca, X_ca, num_jac_perturbs, rel_jac_perturb)
    # else
    #     _J_ca = J_ca
    # end

    # # So, first thing is to get the sparsity.
    # J_sparse = ADTypes.jacobian_sparsity(compute_adable!, Y_ca, X_ca, ADTypes.sparsity_detector(AD))

    # # Create a ComponentMatrix version of J_sparse.
    # J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # # Get some colors!
    # colors = matrix_colors(getdata(J_ca_sparse))

    # # Create the cache object for `forwarddiff_color_jacobian!`.
    # jac_cache = ForwardColorJacCache(f!, X_ca; dx=Y_ca, colorvec=colors, sparsity=getdata(J_ca_sparse))

    # Need to "prepare" the backend.
    prep = DifferentiationInterface.prepare_jacobian(compute_adable!, Y_ca, ad, X_ca)

    # Now I think I can get the sparse Jacobian from that.
    J_sparse = Float64.(SparseMatrixColorings.sparsity_pattern(prep))

    # Then use that sparse Jacobian to create the component matrix version.
    J_ca_sparse = ComponentMatrix(J_sparse, (only(getaxes(Y_ca,)), only(getaxes(X_ca))))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Create complex-valued versions of the X_ca and Y_ca arrays.
    X_ca_cs = similar(X_ca, ComplexF64)
    Y_ca_cs = similar(Y_ca, ComplexF64)

    return ADSparseExplicitComp(ad, compute_adable!, X_ca, Y_ca, J_ca_sparse, prep, rcdict, units_dict, tags_dict, X_ca_cs, Y_ca_cs)
end

get_callback(comp::ADSparseExplicitComp) = comp.compute_adable!

get_input_ca(::Type{Float64}, comp::ADSparseExplicitComp) = comp.X_ca
get_input_ca(::Type{ComplexF64}, comp::ADSparseExplicitComp) = comp.X_ca_cs
get_input_ca(::Type{Any}, comp::ADSparseExplicitComp) = comp.X_ca
get_input_ca(comp::ADSparseExplicitComp) = get_input_ca(Float64, comp)

get_output_ca(::Type{Float64}, comp::ADSparseExplicitComp) = comp.Y_ca
get_output_ca(::Type{ComplexF64}, comp::ADSparseExplicitComp) = comp.Y_ca_cs
get_output_ca(::Type{Any}, comp::ADSparseExplicitComp) = comp.Y_ca
get_output_ca(comp::ADSparseExplicitComp) = get_output_ca(Float64, comp)

get_sparse_jacobian_ca(comp::ADSparseExplicitComp) = comp.J_ca_sparse
get_prep(comp::ADSparseExplicitComp) = comp.prep
get_units(comp::ADSparseExplicitComp, varname) = get(comp.units_dict, varname, "unitless")
get_tags(comp::ADSparseExplicitComp, varname) = get(comp.tags_dict, varname, Vector{String}())
get_rows_cols_dict(comp::ADSparseExplicitComp) = comp.rcdict
get_backend(comp::ADSparseExplicitComp) = comp.ad

function get_input_var_data(self::ADSparseExplicitComp)
    ca = get_input_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
end

function get_output_var_data(self::ADSparseExplicitComp)
    ca = get_output_ca(self)
    return [VarData(string(k); shape=size(ca[k]), val=ca[k], units=get_units(self, k), tags=get_tags(self, k)) for k in keys(ca)]
end

function get_partials_data(self::ADSparseExplicitComp)
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

function OpenMDAOCore.setup(self::ADSparseExplicitComp)
    input_data = get_input_var_data(self)
    output_data = get_output_var_data(self)
    partials_data = get_partials_data(self)

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ADSparseExplicitComp, inputs, outputs)
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

function OpenMDAOCore.compute_partials!(self::ADSparseExplicitComp, inputs, partials)
    # Copy the inputs into the input `ComponentArray`.
    X_ca = get_input_ca(self)
    for iname in keys(X_ca)
        # This works even if `X_ca[iname]` is a scalar, because of the `@view`!
        @view(X_ca[iname]) .= inputs[string(iname)]
    end

    # Get the Jacobian.
    f! = get_callback(self)
    Y_ca = get_output_ca(self)
    J_ca_sparse = get_sparse_jacobian_ca(self)
    prep = get_prep(self)
    backend = get_backend(self)
    DifferentiationInterface.jacobian!(f!, Y_ca, J_ca_sparse, prep, backend, X_ca)

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
            Jsub_out = partials[string(oname), string(iname)]

            # This will get a vector of the non-zero entries of the sparse sub-Jacobian if it's actually sparse, or just a reference to the flattened vector of the dense sub-Jacobian otherwise.
            Jsub_out_vec = _maybe_nonzeros(Jsub_out)

            # Now write the non-zero entries to Jsub_out_vec.
            Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
        end
    end

    return nothing
end

has_setup_partials(self::ADSparseExplicitComp) = false
has_compute_partials(self::ADSparseExplicitComp) = true
has_compute_jacvec_product(self::ADSparseExplicitComp) = false
