using OpenMDAOCore
using Test
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using SparseArrays: sparse, findnz, nnz, issparse
using SparseDiffTools: matrix_colors, ForwardColorJacCache

struct Comp1{TCompute,TX,TY,TJ,TCache,TRCDict} <: AbstractAutoSparseForwardDiffExplicitComp #where {TCompute,TX,TY,TJ,TCache,TRCDict}
    compute_forwarddiffable!::TCompute
    X_ca::TX
    Y_ca::TY
    J_ca_sparse::TJ
    jac_cache::TCache
    rcdict::TRCDict
end

function Comp1(M, N)
    compute_forwarddiffable! = let M=M, N=N
        (Y, X)->begin
            a = only(X[:a])
            b = @view X[:b]
            c = @view X[:c]
            d = @view X[:d]
            e = @view Y[:e]
            f = @view Y[:f]
            g = @view Y[:g]

            for n in 1:N
                e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
                for m in 1:M
                    f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
                    g[n, m] = 10*sin(b[n])*cos(d[m, n])
                end
            end
            return nothing
        end
    end
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # Create a dense ComponentMatrix from the input and output arrays.
    J_ca = Y_ca.*X_ca'

    # Define the sparsity by writing ones and zeros to the J_ca dense `ComponentMatrix`.
    J_ca .= 0.0
    for n in 1:N
        @view(J_ca[:e, :a])[n] = 1.0
        @view(J_ca[:e, :b])[n, n] = 1.0
        for m in 1:M
            @view(J_ca[:e, :c])[n, m] = 1.0
            @view(J_ca[:e, :d])[n, m, n] = 1.0

            @view(J_ca[:f, :a])[m, n] = 1.0
            @view(J_ca[:f, :b])[m, n, n] = 1.0
            @view(J_ca[:f, :c])[m, n, m] = 1.0
            @view(J_ca[:f, :d])[m, n, m, n] = 1.0

            @view(J_ca[:g, :b])[n, m, n] = 1.0
            @view(J_ca[:g, :d])[n, m, m, n] = 1.0
        end
    end

    # Create a sparse matrix version of J_ca.
    J_ca_sparse = ComponentMatrix(sparse(getdata(J_ca)), getaxes(J_ca))

    # Get a dictionary describing the non-zero rows and cols for each subjacobian.
    rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Get some colors!
    colors = matrix_colors(getdata(J_ca_sparse))

    # Create the cache object for `forwarddiff_color_jacobian!`.
    jac_cache = ForwardColorJacCache(compute_forwarddiffable!, X_ca; dx=Y_ca, colorvec=colors, sparsity=getdata(J_ca_sparse))

    return Comp1(compute_forwarddiffable!, X_ca, Y_ca, J_ca_sparse, jac_cache, rcdict)
end

# Don't worry about units and tags for now.
get_units(self::Comp1, varname) = nothing
get_tags(self::Comp1, varname) = nothing

function doit()
    # Create the component.
    N = 3
    M = 4
    comp = Comp1(M, N)

    rcdict = get_rows_cols_dict(comp)

    inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict["a"] .= 2.0
    inputs_dict["b"] .= range(3.0, 4.0; length=N)
    inputs_dict["c"] .= range(5.0, 6.0; length=M)
    inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp))

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict["e"] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict["f"] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict["g"] .≈ g_check)

    J_ca_sparse = get_sparse_jacobian_ca(comp)
    @test issparse(getdata(J_ca_sparse))
    @test size(getdata(J_ca_sparse)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
    @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    partials_dict = rcdict2strdict(rcdict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    rows, cols = rcdict[:e, :a]
    vals = partials_dict["e", "a"]
    @test size(vals) == (N,)
    deda_check = zeros(N)
    for n in 1:N
        deda_check[n] = 4*only(a)
    end
    deda_check_sparse = sparse(reshape(deda_check, N))
    # `e` is a vector of length `N` and `a` is scalar, so the Jacobian isn't actually a Matrix (and isn't really sparse).
    rows_check, vals_check = findnz(deda_check_sparse)
    cols_check = fill(1, N)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:e, :b]
    vals = partials_dict["e", "b"]
    @test size(vals) == (N,)
    dedb_check = zeros(N, N)
    for n in 1:N
        dedb_check[n, n] = (3*2.1)*b[n]^1.1
    end
    dedb_check_sparse = sparse(reshape(dedb_check, N, N))
    rows_check, cols_check, vals_check = findnz(dedb_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:e, :c]
    vals = partials_dict["e", "c"]
    @test size(vals) == (N*M,)
    dedc_check = zeros(N, M)
    for m in 1:M
        for n in 1:N
            dedc_check[n, m] = (4*2.2)*c[m]^1.2
        end
    end
    dedc_check_sparse = sparse(reshape(dedc_check, N, M))
    rows_check, cols_check, vals_check = findnz(dedc_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:e, :d]
    vals = partials_dict["e", "d"]
    @test size(vals) == (M*N,)
    dedd_check = zeros(N, M, N)
    for n in 1:N
        for m in 1:M
            dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
        end
    end
    dedd_check_sparse = sparse(reshape(dedd_check, N, M*N))
    rows_check, cols_check, vals_check = findnz(dedd_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:f, :a]
    vals = partials_dict["f", "a"]
    @test size(vals) == (M*N,)
    dfda_check = zeros(M, N)
    for m in 1:M
        for n in 1:N
            dfda_check[m, n] = (6*2.4)*only(a)^1.4
        end
    end
    dfda_check_sparse = sparse(reshape(dfda_check, M*N, 1))
    rows_check, cols_check, vals_check = findnz(dfda_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:f, :b]
    vals = partials_dict["f", "b"]
    @test size(vals) == (M*N,)
    dfdb_check = zeros(M, N, N)
    for n in 1:N
        for m in 1:M
            dfdb_check[m, n, n] = (7*2.5)*b[n]^1.5
        end
    end
    dfdb_check_sparse = sparse(reshape(dfdb_check, M*N, N))
    rows_check, cols_check, vals_check = findnz(dfdb_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:f, :c]
    vals = partials_dict["f", "c"]
    @test size(vals) == (M*N,)
    dfdc_check = zeros(M, N, M)
    for n in 1:N
        for m in 1:M
            dfdc_check[m, n, m] = (8*2.6)*c[m]^1.6
        end
    end
    dfdc_check_sparse = sparse(reshape(dfdc_check, M*N, M))
    rows_check, cols_check, vals_check = findnz(dfdc_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:f, :d]
    vals = partials_dict["f", "d"]
    @test size(vals) == (M*N,)
    dfdd_check = zeros(M, N, M, N)
    for n in 1:N
        for m in 1:M
            dfdd_check[m, n, m, n] = (9*2.7)*d[m, n]^1.7
        end
    end
    dfdd_check_sparse = sparse(reshape(dfdd_check, M*N, M*N))
    rows_check, cols_check, vals_check = findnz(dfdd_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:g, :a]
    vals = partials_dict["g", "a"]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64

    rows, cols = rcdict[:g, :b]
    vals = partials_dict["g", "b"]
    @test size(vals) == (N*M,)
    dgdb_check = zeros(N, M, N)
    for m in 1:M
        for n in 1:N
            dgdb_check[n, m, n] = 10*cos(b[n])*cos(d[m, n])
        end
    end
    dgdb_check_sparse = sparse(reshape(dgdb_check, N*M, N))
    rows_check, cols_check, vals_check = findnz(dgdb_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    rows, cols = rcdict[:g, :c]
    vals = partials_dict["g", "c"]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64

    rows, cols = rcdict[:g, :d]
    vals = partials_dict["g", "d"]
    @test size(vals) == (N*M,)
    dgdd_check = zeros(N, M, M, N)
    for m in 1:M
        for n in 1:N
            dgdd_check[n, m, m, n] = -10*sin(b[n])*sin(d[m, n])
        end
    end
    dgdd_check_sparse = sparse(reshape(dgdd_check, N*M, M*N))
    rows_check, cols_check, vals_check = findnz(dgdd_check_sparse)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
    @test all(vals .≈ vals_check)

    # Check that the partials_dict created by ca2strdict gives the same result as one created using rcdict2strdict.
    partials_dict2 = ca2strdict(J_ca_sparse)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

end

doit()
