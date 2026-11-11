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

# Moving this to the sparse extension.
function ca2strdict_sparse end
function _get_rows_cols_dict_from_sparsity end

# AD backend capability helpers (ported from OpenMDAO4Core.jl).
# Used by `create_explicit_component` to validate that a backend supports the
# derivative mode implied by the chosen flavor (forward -> JVP, reverse -> VJP).
function can_jvp(adtype)
    adtype_mode = ADTypes.mode(adtype)
    return (adtype_mode isa ADTypes.ForwardMode) || (adtype_mode isa ADTypes.ForwardOrReverseMode)
end

function can_vjp(adtype)
    adtype_mode = ADTypes.mode(adtype)
    return (adtype_mode isa ADTypes.ReverseMode) || (adtype_mode isa ADTypes.ForwardOrReverseMode)
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

# ---------------------------------------------------------------------------
# Prepared copying between `Dict`s and `ComponentVector`/`ComponentMatrix`
# storage (ported from OpenMDAO4Core.jl).
#
# The copy "preps" hold the per-component names and flat index ranges of a
# `ComponentVector` (or `ComponentMatrix`), computed once at component-creation
# time. The copy functions then operate directly on the flat data via
# `getdata`, avoiding per-call `ComponentArray` axis lookups, view
# construction, and `string` conversions.
# ---------------------------------------------------------------------------

"""
    InputCopyPrep

Prepared metadata for copying string-keyed component variables into the flat
storage of a `ComponentVector` (or out of it).
"""
struct InputCopyPrep
    input_names::Vector{String}
    input_ranges::Vector{UnitRange{Int}}
end

function _component_ranges(ca::ComponentVector)
    axis = only(getaxes(ca))
    names = String[]
    ranges = UnitRange{Int}[]
    for name in keys(axis)
        indices = axis[name]
        range = indices isa Integer ? (indices:indices) : getfield(indices, :idx)
        push!(names, string(name))
        push!(ranges, first(range):last(range))
    end
    return names, ranges
end

InputCopyPrep(ca::ComponentVector) = InputCopyPrep(_component_ranges(ca)...)

"""
    _copy_component_inputs!(data, inputs, names, ranges)

Copy dictionary-backed component variables into flat `ComponentVector` storage.
The source and destination fields are expected to have the same logical shape;
copying by Cartesian index preserves that shape even when their storage layouts
differ.
"""
function _copy_component_inputs!(data, inputs, names, ranges)
    for (name, range) in zip(names, ranges)
        source = inputs[name]
        destination = view(data, range)
        if source isa AbstractArray
            destination_indices = LinearIndices(size(source))
            @inbounds for I in CartesianIndices(source)
                destination[destination_indices[I]] = source[I]
            end
        else
            destination[1] = source
        end
    end
    return nothing
end

_copy_component_inputs!(data, inputs, prep::InputCopyPrep) =
    _copy_component_inputs!(data, inputs, prep.input_names, prep.input_ranges)

"""Copy positionally ordered component variables into prepared flat ranges."""
function _copy_component_inputs!(data, inputs::Tuple, ranges)
    for (source, range) in zip(inputs, ranges)
        destination = view(data, range)
        if source isa AbstractArray
            destination_indices = LinearIndices(size(source))
            @inbounds for I in CartesianIndices(source)
                destination[destination_indices[I]] = source[I]
            end
        else
            destination[1] = source
        end
    end
    return nothing
end

"""
    _copy_component_deriv_inputs!(data, d_inputs, names, ranges)

Copy dictionary-backed derivative (seed) variables into flat `ComponentVector`
storage, filling any variable that is missing from the dictionary with zeros.
Copying is by Cartesian index so that differing storage layouts are handled.
"""
function _copy_component_deriv_inputs!(data, d_inputs, names, ranges)
    for (name, range) in zip(names, ranges)
        destination = view(data, range)
        if name in keys(d_inputs)
            source = d_inputs[name]
            if source isa AbstractArray
                destination_indices = LinearIndices(size(source))
                @inbounds for I in CartesianIndices(source)
                    destination[destination_indices[I]] = source[I]
                end
            else
                destination[1] = source
            end
        else
            fill!(destination, zero(eltype(destination)))
        end
    end
    return nothing
end

_copy_component_deriv_inputs!(data, d_inputs, prep::InputCopyPrep) =
    _copy_component_deriv_inputs!(data, d_inputs, prep.input_names, prep.input_ranges)

"""
    _copy_component_outputs!(data, outputs, names, ranges)

Copy flat component data into dictionary-backed output arrays or scalars. The
logical shape is taken from each destination dictionary entry, so source and
destination storage layouts may differ.
"""
function _copy_component_outputs!(data, outputs, names, ranges)
    for (name, range) in zip(names, ranges)
        source = view(data, range)
        destination = outputs[name]
        if destination isa AbstractArray
            source_indices = LinearIndices(size(destination))
            @inbounds for I in CartesianIndices(destination)
                destination[I] = source[source_indices[I]]
            end
        else
            outputs[name] = only(source)
        end
    end
    return nothing
end

_copy_component_outputs!(data, outputs, prep::InputCopyPrep) =
    _copy_component_outputs!(data, outputs, prep.input_names, prep.input_ranges)

"""
    _copy_component_outputs_accum!(data, outputs, names, ranges)

Accumulate flat component data into dictionary-backed output arrays or scalars.
The logical shape is taken from each destination dictionary entry, so source and
destination storage layouts may differ.
"""
function _copy_component_outputs_accum!(data, outputs, names, ranges)
    for (name, range) in zip(names, ranges)
        source = view(data, range)
        destination = outputs[name]
        if destination isa AbstractArray
            source_indices = LinearIndices(size(destination))
            @inbounds for I in CartesianIndices(destination)
                destination[I] += source[source_indices[I]]
            end
        else
            outputs[name] += only(source)
        end
    end
    return nothing
end

_copy_component_outputs_accum!(data, outputs, prep::InputCopyPrep) =
    _copy_component_outputs_accum!(data, outputs, prep.input_names, prep.input_ranges)

"""
    DensePartialCopyPrep

Prepared metadata for copying the sub-Jacobians of a dense
`ComponentMatrix`-backed Jacobian into a string-tuple-keyed partials
dictionary.
"""
struct DensePartialCopyPrep
    output_names::Vector{String}
    input_names::Vector{String}
    output_ranges::Vector{UnitRange{Int}}
    input_ranges::Vector{UnitRange{Int}}
end

function DensePartialCopyPrep(J_ca::ComponentMatrix)
    row_axis, col_axis = getaxes(J_ca)
    output_names = String[]
    input_names = String[]
    output_ranges = UnitRange{Int}[]
    input_ranges = UnitRange{Int}[]
    for output_name in keys(row_axis), input_name in keys(col_axis)
        output_indices = row_axis[output_name]
        input_indices = col_axis[input_name]
        output_range = output_indices isa Integer ? (output_indices:output_indices) : getfield(output_indices, :idx)
        input_range = input_indices isa Integer ? (input_indices:input_indices) : getfield(input_indices, :idx)
        push!(output_names, string(output_name))
        push!(input_names, string(input_name))
        push!(output_ranges, first(output_range):last(output_range))
        push!(input_ranges, first(input_range):last(input_range))
    end
    return DensePartialCopyPrep(output_names, input_names, output_ranges, input_ranges)
end

"""
    _prepare_sparse_blocks(J_ca, rcdict)

Build a `Vector{SparsePartialBlock}` from a sparse `ComponentMatrix`-backed
Jacobian `J_ca` and a row/column index dictionary `rcdict` mapping each
`(output_name, input_name)` sub-Jacobian to its 1-based nonzero row and column
indices (relative to the sub-Jacobian). The block rows/cols are converted to
global (whole-Jacobian) indices so that values can be scattered without any
per-call sub-Jacobian lookups.
"""
function _prepare_sparse_blocks(J_ca::ComponentMatrix, rcdict)
    row_axis, col_axis = getaxes(J_ca)
    row_starts = Dict{Symbol,Int}()
    col_starts = Dict{Symbol,Int}()
    for name in keys(row_axis)
        indices = row_axis[name]
        range = indices isa Integer ? (indices:indices) : getfield(indices, :idx)
        row_starts[name] = first(range)
    end
    for name in keys(col_axis)
        indices = col_axis[name]
        range = indices isa Integer ? (indices:indices) : getfield(indices, :idx)
        col_starts[name] = first(range)
    end

    blocks = SparsePartialBlock[]
    for ((output_name, input_name), (rows, cols)) in rcdict
        global_rows = [row_starts[output_name] + row - 1 for row in rows]
        global_cols = [col_starts[input_name] + col - 1 for col in cols]
        push!(blocks, SparsePartialBlock(
            string(output_name),
            string(input_name),
            global_rows,
            global_cols,
        ))
    end
    return blocks
end

# Sparse partials assembly: core declares the function; the sparse extension
# (which knows about `SparseMatrixCSC` specifics) provides the implementation.
function _assemble_sparse_partials! end

"""
    _copy_dense_partials!(partials, J_ca, prep)

Scatter the sub-Jacobians of the dense `ComponentMatrix`-backed Jacobian
`J_ca` into the string-tuple-keyed `partials` mapping, reshaping each
sub-Jacobian as needed. Keys not present in `partials` (i.e., OpenMDAO did not
ask for them) are skipped.
"""
function _copy_dense_partials!(partials, J_ca, prep::DensePartialCopyPrep)
    J_data = getdata(J_ca)
    for i in eachindex(prep.output_names)
        local partial
        try
            partial = partials[prep.output_names[i], prep.input_names[i]]
        catch e
            isa(e, KeyError) || rethrow()
        else
            source = @view J_data[prep.output_ranges[i], prep.input_ranges[i]]
            partial .= reshape(source, size(partial))
        end
    end
    return nothing
end

# Dense fallback for `_maybe_nonzeros`: return the array unchanged.
# Sparse-specific overloads (for `AbstractSparseArray` and reshaped sparse views)
# are provided by the `OpenMDAOCoreSparseMatrixColoringsExt` extension.
_maybe_nonzeros(A::AbstractArray) = A


function _resize_component_vector(X_ca, sizes)
    # Preserve the original key order of `X_ca` by building a `NamedTuple`.
    # A `NamedTuple`'s field order is part of its type, so it is guaranteed to
    # be preserved when constructing a `ComponentVector` from it (unlike a plain
    # `Dict`, whose iteration order is unspecified). This replaces the previous
    # `OrderedDict`-based implementation, removing the `DataStructures` dep.
    ks = collect(keys(X_ca))
    vals = map(ks) do ca_name
        if ca_name in keys(sizes)
            # Create an array of the appropriate size.
            new_arr = zeros(eltype(X_ca), sizes[ca_name])
            # Fill it with the value it should have.
            new_arr .= X_ca[ca_name]
            return new_arr
        else
            return X_ca[ca_name]
        end
    end
    nt = NamedTuple{tuple(ks...)}(tuple(vals...))
    return ComponentVector(nt)
end

# ---------------------------------------------------------------------------
# PerturbedDenseSparsityDetector
#
# The *type* (struct + `show` + constructor) is declared here in the main
# package so it can be imported without the extension loaded. Only the
# sparsity-detection *methods* (`ADTypes.jacobian_sparsity`/
# `hessian_sparsity`/`jacobian_sparsity!`) require `SparseArrays` and live in the
# `OpenMDAOCoreSparseMatrixColoringsExt` extension.
# ---------------------------------------------------------------------------

"""
    PerturbedDenseSparsityDetector

Tweaked version of [`DenseSparsityDetector`](https://gdalle.github.io/DifferentiationInterface.jl/DifferentiationInterface/stable/api/#DifferentiationInterface.DenseSparsityDetector) sparsity pattern detector satisfying the [detection API](https://sciml.github.io/ADTypes.jl/stable/#Sparse-AD) of [ADTypes.jl](https://github.com/SciML/ADTypes.jl) that evaluates the Jacobian multiple times using a perturbed input vector.
Specifically, input vector `x` will be perturbed via

```julia
    x_perturb = (1 .+ rel_x_perturb.*perturb1).*x .+ perturb2.*abs_x_perturb
```

where `perturb1` and `perturb2` are random `Vector`s of numbers ranging from `-0.5` to `0.5`, and `rel_x_perturb` and `abs_x_perturb` are relative and absolute perturbation magnitudes specified by the user.

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

    PerturbedDenseSparsityDetector(backend; atol, method=:iterative, nevals=3, rel_x_perturb=0.001, abs_x_perturb=0.0001)

The keyword argument `method::Symbol` can be either:

- `:iterative`: compute the matrix in a sequence of matrix-vector products (memory-efficient)
- `:direct`: compute the matrix all at once (memory-hungry but sometimes faster).

Note that the constructor is type-unstable because `method` ends up being a type parameter of the `PerturbedDenseSparsityDetector` object (this is not part of the API and might change).

"""
struct PerturbedDenseSparsityDetector{method,B,TRelXPerturb,TAbsXPerturb} <: ADTypes.AbstractSparsityDetector
    backend::B
    atol::Float64
    nevals::Int
    rel_x_perturb::TRelXPerturb
    abs_x_perturb::TAbsXPerturb
end

function Base.show(io::IO, detector::PerturbedDenseSparsityDetector{method}) where {method}
    (; backend, atol, nevals, rel_x_perturb, abs_x_perturb) = detector
    return print(
        io,
        PerturbedDenseSparsityDetector,
        "(",
        repr(backend; context=io),
        "; atol=$atol, method=",
        repr(method; context=io),
        "nevals=$nevals, rel_x_perturb=$rel_x_perturb", "abs_x_perturb=$abs_x_perturb",
        ")",
    )
end

function PerturbedDenseSparsityDetector(
    backend::ADTypes.AbstractADType; atol::Float64, method::Symbol=:iterative, nevals=3, rel_x_perturb=0.001, abs_x_perturb=0.0001
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

    return PerturbedDenseSparsityDetector{method,typeof(backend),typeof(rel_x_perturb),typeof(abs_x_perturb)}(backend, atol, nevals, rel_x_perturb, abs_x_perturb)
end
