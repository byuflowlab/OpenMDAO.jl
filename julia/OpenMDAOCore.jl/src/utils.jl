"""
    get_rows_cols(; ss_sizes::Dict{Symbol, Int}, of_ss::AbstractVector{Symbol}, wrt_ss::AbstractVector{Symbol})

Get the non-zero row and column indices for a sparsity pattern defined by output subscripts `of_ss` and input subscripts `wrt_ss`.

`ss_sizes` is a `Dict` mapping the subscript symbols in `of_ss` and `wrt_ss` to the size of each dimension the subscript symbols correspond to.
The returned indices will be zero-based, which is what the OpenMDAO `declare_partials` method expects.

# Examples
Diagonal partials for 1D output and 1D input, both with length `5`:
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>5), of_ss=[:i], wrt_ss=[:i])
([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
```

1D output with length 2 depending on all elements of 1D input with length 3 (so not actually sparse).
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>2, :j=>3), of_ss=[:i], wrt_ss=[:j])
([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
```

2D output with size `(2, 3)` and 1D input with size `2`, where each `i` output row only depends on the `i` input element.
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>2, :j=>3), of_ss=[:i, :j], wrt_ss=[:i])
([0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1])
```

2D output with size `(2, 3)` and 1D input with size `3`, where each `j` output column only depends on the `j` input element.
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>2, :j=>3), of_ss=[:i, :j], wrt_ss=[:j])
([0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2])
```

2D output with size `(2, 3)` depending on input with size `(3, 2)`, where the output element at index `i, j` only depends on input element `j, i` (like a transpose operation).
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>2, :j=>3), of_ss=[:i, :j], wrt_ss=[:j, :i])
([0, 1, 2, 3, 4, 5], [0, 2, 4, 1, 3, 5])
```

2D output with size `(2, 3)` depending on input with size `(3, 4)`, where output `y[:, j]` for each `j` depends on input `x[j, :]`.
```jldoctest; setup = :(using OpenMDAOCore: get_rows_cols)
julia> rows, cols = get_rows_cols(; ss_sizes=Dict(:i=>2, :j=>3, :k=>4), of_ss=[:i, :j], wrt_ss=[:j, :k]);

julia> @show rows cols;  # to prevent abbreviating the array display
rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

```
"""
function get_rows_cols(; ss_sizes, of_ss, wrt_ss, column_major=true, zero_based_indexing=true)
    # Get the output subscript, which will start with the of_ss, then the
    # wrt_ss with the subscripts common to both removed.
    # deriv_ss = of_ss + "".join(set(wrt_ss) - set(of_ss))
    deriv_ss = vcat(of_ss, setdiff(wrt_ss, of_ss))

    if column_major
        # Reverse the subscripts so they work with column-major ordering.
        of_ss = reverse(of_ss)
        wrt_ss = reverse(wrt_ss)
        deriv_ss = reverse(deriv_ss)
    end

    # Get the shape of the output variable (the "of"), the input variable
    # (the "wrt"), and the derivative (the Jacobian).
    of_shape = Tuple(ss_sizes[s] for s in of_ss)
    wrt_shape = Tuple(ss_sizes[s] for s in wrt_ss)
    deriv_shape = Tuple(ss_sizes[s] for s in deriv_ss)

    # Invert deriv_ss: get a dictionary that goes from subscript to index
    # dimension.
    deriv_ss2idx = Dict(ss=>i for (i, ss) in enumerate(deriv_ss))

    # This is the equivalent of the Python code
    #   a = np.arange(np.prod(of_shape)).reshape(of_shape)
    #   b = np.arange(np.prod(wrt_shape)).reshape(wrt_shape)
    # but in column major order, which is OK, since we've reversed the order of
    # of_shape and wrt_shape above.
    a = reshape(0:prod(of_shape)-1, of_shape)
    b = reshape(0:prod(wrt_shape)-1, wrt_shape)

    # If not using zero-based indexing, adjust for that by adding one to everything.
    if !zero_based_indexing
        a = a .+ 1
        b = b .+ 1
    end

    rows = Array{Int}(undef, deriv_shape)
    cols = Array{Int}(undef, deriv_shape)
    for deriv_idx in CartesianIndices(deriv_shape)
        # Go from the jacobian index to the of and wrt indices.
        of_idx = [deriv_idx[deriv_ss2idx[ss]] for ss in of_ss]
        wrt_idx = [deriv_idx[deriv_ss2idx[ss]] for ss in wrt_ss]

        # Get the flattened index for the output and input.
        rows[deriv_idx] = a[of_idx...]
        cols[deriv_idx] = b[wrt_idx...]
    end

    # Return flattened versions of the rows and cols arrays.
    return rows[:], cols[:]
end


"""
    get_rows_cols_dict_from_sparsity(J::ComponentMatrix)

Get a `Dict` of the non-zero row and column indices for a sparsity pattern defined by a `ComponentMatrix` representation of a Jacobian.
"""
function get_rows_cols_dict_from_sparsity(J::ComponentMatrix)
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

_at_least_1d(x) = [x]
_at_least_1d(x::AbstractArray) = x

function ca2strdict(ca::ComponentVector)
    # Might be faster using `valkeys` instead of `keys`.
    return Dict(string(k)=>_at_least_1d(ca[k]) for k in keys(ca))
end

_at_least_2d(x) = [x;;]
# Not sure what to do about the case when `x` is `<:AbstractVector`.
# Could add a one to the size, but at the beginning or end?
_at_least_2d(x::AbstractArray) = x

function ca2strdict(ca::ComponentMatrix)
    raxis, caxis = getaxes(ca)
    return Dict((string(rname), string(cname))=>_at_least_2d(ca[rname, cname]) for rname in keys(raxis), cname in keys(caxis))
end

function ca2strdict_sparse(ca::ComponentMatrix)
    T = eltype(ca)
    raxis, caxis = getaxes(ca)
    out = Dict{Tuple{String,String}, Vector{T}}()
    for input_name in keys(caxis)
        for output_name in keys(raxis)
            # @show ca[output_name, input_name]
            Jsub = ca[output_name, input_name]
            Jsub_reshape = reshape(Jsub, length(raxis[output_name]), length(caxis[input_name]))
            data_sparse = sparse(Jsub_reshape)
            out[string(output_name), string(input_name)] = nonzeros(data_sparse)
        end
    end
    return out
end

function rcdict2strdict(::Type{T}, rcdict) where {T}
    out = Dict{Tuple{String,String}, Vector{T}}()
    for (output_name, input_name) in keys(rcdict)
        rows, cols = rcdict[output_name, input_name]
        out[string(output_name), string(input_name)] = zeros(T, length(rows))
    end
    return out
end
rcdict2strdict(rcdict) = rcdict2strdict(Float64, rcdict)

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
            # prep = DifferentiationInterface.prepare_pushforward_same_point(
            #     f, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),)
            # )
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                # DifferentiationInterface.pushforward!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
                DifferentiationInterface.pushforward!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            # prep = DifferentiationInterface.prepare_pullback_same_point(
            #     f, backend, x_perturb, (DifferentiationInterface.basis(backend, y, first(eachindex(y))),)
            # )
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f, backend, x_perturb, (DifferentiationInterface.basis(y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                # DifferentiationInterface.pullback!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, y, i),))
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
            # prep = DifferentiationInterface.prepare_pushforward_same_point(
            #     f!, y, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),)
            # )
            prep = DifferentiationInterface.prepare_pushforward_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),)
            )
            for (kj, j) in enumerate(eachindex(x_perturb))
                # DifferentiationInterface.pushforward!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
                DifferentiationInterface.pushforward!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, j),))
                for ki in LinearIndices(p)
                    if (abs(p[ki]) > atol) && !((ki, kj) in IJ)
                        push!(IJ, (ki, kj))
                    end
                end
            end
        else
            p = similar(x_perturb)
            # prep = DifferentiationInterface.prepare_pullback_same_point(
            #     f!, y, backend, x_perturb, (DifferentiationInterface.basis(backend, y, first(eachindex(y))),)
            # )
            prep = DifferentiationInterface.prepare_pullback_same_point(
                f!, y, backend, x_perturb, (DifferentiationInterface.basis(y, first(eachindex(y))),)
            )
            for (ki, i) in enumerate(eachindex(y))
                # DifferentiationInterface.pullback!(f!, y, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, y, i),))
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

        # prep = DifferentiationInterface.prepare_hvp_same_point(f, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, first(eachindex(x_perturb))),))
        prep = DifferentiationInterface.prepare_hvp_same_point(f, backend, x_perturb, (DifferentiationInterface.basis(x_perturb, first(eachindex(x_perturb))),))
        for (kj, j) in enumerate(eachindex(x_perturb))
            # hvp!(f, (p,), prep, backend, x_perturb, (DifferentiationInterface.basis(backend, x_perturb, j),))
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

