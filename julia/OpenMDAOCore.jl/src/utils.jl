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
    for output_name in keys(raxis)
        for input_name in keys(caxis)
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
