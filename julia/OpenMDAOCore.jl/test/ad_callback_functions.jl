function f_simple!(Y, X, params)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]
    e = @view Y[:e]
    f = @view Y[:f]
    g = @view Y[:g]

    M, N = params
    for n in 1:N
        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
        for m in 1:M
            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
            g[n, m] = 10*sin(b[n])*cos(d[m, n])
        end
    end

    return nothing
end

function f_simple(X, params)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]

    e = (2*a^2) .+ 3.0.*b.^2.1 .+ 4.0.*sum(c.^2.2) .+ 5.0.*vec(sum(d.^2.3; dims=1))
    f = (6*a^2.4) .+ 7.0.*reshape(b, 1, :).^2.5 .+ 8.0.*c.^2.6 .+ 9.0.*d.^2.7
    g = 10.0.*sin.(b).*cos.(PermutedDimsArray(d, (2, 1)))

    Y = ComponentVector(e=e, f=f, g=g)
    return Y
end

function f_simple_no_params!(Y, X, params)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]
    e = @view Y[:e]
    f = @view Y[:f]
    g = @view Y[:g]

    M, N = size(f)
    for n in 1:N
        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
        for m in 1:M
            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
            g[n, m] = 10*sin(b[n])*cos(d[m, n])
        end
    end

    return nothing
end

function do_compute_check(comp, aviary_input_vars=Dict{Symbol,Dict}(), aviary_output_vars=Dict{Symbol,Dict}())
    aviary_input_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_input_vars)
    aviary_output_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_output_vars)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    M, N = size(inputs_dict[get_aviary_input_name(comp, :d)])
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
    inputs_dict[get_aviary_input_name(comp, :b)] .= range(3.0, 4.0; length=N)
    inputs_dict[get_aviary_input_name(comp, :c)] .= range(5.0, 6.0; length=M)
    inputs_dict[get_aviary_input_name(comp, :d)] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    return nothing
end

function do_compute_partials_check(comp, aviary_input_vars=Dict{Symbol,Dict}(), aviary_output_vars=Dict{Symbol,Dict}())
    sparse_jac = typeof(comp) <: SparseADExplicitComp

    aviary_input_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_input_vars)
    aviary_output_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_output_vars)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    M, N = size(inputs_dict[get_aviary_input_name(comp, :d)])
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
    inputs_dict[get_aviary_input_name(comp, :b)] .= range(3.0, 4.0; length=N)
    inputs_dict[get_aviary_input_name(comp, :c)] .= range(5.0, 6.0; length=M)
    inputs_dict[get_aviary_input_name(comp, :d)] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)

    inputs_dict_cs = ca2strdict(get_input_ca(ComplexF64, comp), aviary_input_names)
    inputs_dict_cs[get_aviary_input_name(comp, :a)] .= inputs_dict[get_aviary_input_name(comp, :a)]
    inputs_dict_cs[get_aviary_input_name(comp, :b)] .= inputs_dict[get_aviary_input_name(comp, :b)]
    inputs_dict_cs[get_aviary_input_name(comp, :c)] .= inputs_dict[get_aviary_input_name(comp, :c)]
    inputs_dict_cs[get_aviary_input_name(comp, :d)] .= inputs_dict[get_aviary_input_name(comp, :d)]
    outputs_dict_cs = ca2strdict(get_output_ca(ComplexF64, comp), aviary_output_names)

    # Complex step size.
    h = 1e-10

    J_ca = get_jacobian_ca(comp)

    @test size(getdata(J_ca)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
    if sparse_jac
        @test issparse(getdata(J_ca))
        @test nnz(getdata(J_ca)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    end

    if sparse_jac
        rcdict = get_rows_cols_dict(comp)
        partials_dict = rcdict2strdict(rcdict; cnames=aviary_input_names, rnames=aviary_output_names)
    else
        partials_dict = ca2strdict(J_ca, aviary_input_names, aviary_output_names)
    end

    # Actually do the compute_partials.
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e, f, g = getindex.(Ref(outputs_dict), [get_aviary_output_name(comp, :e), get_aviary_output_name(comp, :f), get_aviary_output_name(comp, :g)])

    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :a)]
    @test size(vals) == (N,)
    deda_check = zeros(N)
    for n in 1:N
        deda_check[n] = 4*only(a)
    end
    if sparse_jac
        rows, cols = rcdict[:e, :a]
        deda_check_sparse = sparse(reshape(deda_check, N))
        # `e` is a vector of length `N` and `a` is scalar, so the Jacobian isn't actually a Matrix (and isn't really sparse).
        rows_check, vals_check = findnz(deda_check_sparse)
        cols_check = fill(1, N)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        vals_check = deda_check
    end
    @test all(vals .≈ vals_check)

    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ deda_check[n]
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    dedb_check = zeros(N, N)
    for n in 1:N
        dedb_check[n, n] = (3*2.1)*b[n]^1.1
    end
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :b)]
    if sparse_jac
        @test size(vals) == (N,)
        rows, cols = rcdict[:e, :b]
        dedb_check_sparse = sparse(reshape(dedb_check, N, N))
        rows_check, cols_check, vals_check = findnz(dedb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(e)..., size(b)...)
        vals_check = dedb_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedb_check[n, n]
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    dedc_check = zeros(N, M)
    for m in 1:M
        for n in 1:N
            dedc_check[n, m] = (4*2.2)*c[m]^1.2
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :c)]
    if sparse_jac
        @test size(vals) == (N*M,)
        rows, cols = rcdict[:e, :c]
        dedc_check_sparse = sparse(reshape(dedc_check, N, M))
        rows_check, cols_check, vals_check = findnz(dedc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(e)..., size(c)...)
        vals_check = dedc_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedc_check[n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    dedd_check = zeros(N, M, N)
    for n in 1:N
        for m in 1:M
            dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :d)]
    if sparse_jac
        @test size(vals) == (M*N,)
        rows, cols = rcdict[:e, :d]
        dedd_check_sparse = sparse(reshape(dedd_check, N, M*N))
        rows_check, cols_check, vals_check = findnz(dedd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(e)..., size(d)...)
        vals_check = dedd_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedd_check[n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    dfda_check = zeros(M, N)
    for m in 1:M
        for n in 1:N
            dfda_check[m, n] = (6*2.4)*only(a)^1.4
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :a)]
    if sparse_jac
        @test size(vals) == (M*N,)
        rows, cols = rcdict[:f, :a]
        dfda_check_sparse = sparse(reshape(dfda_check, M*N, 1))
        rows_check, cols_check, vals_check = findnz(dfda_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == size(f)
        vals_check = dfda_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        for m in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfda_check[m, n]
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    dfdb_check = zeros(M, N, N)
    for n in 1:N
        for m in 1:M
            dfdb_check[m, n, n] = (7*2.5)*b[n]^1.5
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :b)]
    if sparse_jac
        @test size(vals) == (M*N,)
        rows, cols = rcdict[:f, :b]
        dfdb_check_sparse = sparse(reshape(dfdb_check, M*N, N))
        rows_check, cols_check, vals_check = findnz(dfdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(f)..., size(b)...)
        vals_check = dfdb_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdb_check[m, n, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    dfdc_check = zeros(M, N, M)
    for n in 1:N
        for m in 1:M
            dfdc_check[m, n, m] = (8*2.6)*c[m]^1.6
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :c)]
    if sparse_jac
        @test size(vals) == (M*N,)
        rows, cols = rcdict[:f, :c]
        dfdc_check_sparse = sparse(reshape(dfdc_check, M*N, M))
        rows_check, cols_check, vals_check = findnz(dfdc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(f)..., size(c)...)
        vals_check = dfdc_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdc_check[m, n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    dfdd_check = zeros(M, N, M, N)
    for n in 1:N
        for m in 1:M
            dfdd_check[m, n, m, n] = (9*2.7)*d[m, n]^1.7
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :d)]
    if sparse_jac
        @test size(vals) == (M*N,)
        rows, cols = rcdict[:f, :d]
        dfdd_check_sparse = sparse(reshape(dfdd_check, M*N, M*N))
        rows_check, cols_check, vals_check = findnz(dfdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(f)..., size(d)...)
        vals_check = dfdd_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdd_check[m, n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :a)]
    if sparse_jac
        @test size(vals) == (0,)
        rows, cols = rcdict[:g, :a]
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64
    else
        @test size(vals) == size(g)
        @test all(vals .≈ 0)
    end
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for m in 1:M
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    dgdb_check = zeros(N, M, N)
    for m in 1:M
        for n in 1:N
            dgdb_check[n, m, n] = 10*cos(b[n])*cos(d[m, n])
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :b)]
    if sparse_jac
        @test size(vals) == (N*M,)
        rows, cols = rcdict[:g, :b]
        dgdb_check_sparse = sparse(reshape(dgdb_check, N*M, N))
        rows_check, cols_check, vals_check = findnz(dgdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(g)..., size(b)...)
        vals_check = dgdb_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdb_check[n, m, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :c)]
    if sparse_jac
        @test size(vals) == (0,)
        rows, cols = rcdict[:g, :c]
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64
    else
        @test size(vals) == (size(g)..., size(c)...)
        @test all(vals .≈ 0)
    end
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    dgdd_check = zeros(N, M, M, N)
    for m in 1:M
        for n in 1:N
            dgdd_check[n, m, m, n] = -10*sin(b[n])*sin(d[m, n])
        end
    end
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :d)]
    if sparse_jac
        @test size(vals) == (N*M,)
        rows, cols = rcdict[:g, :d]
        dgdd_check_sparse = sparse(reshape(dgdd_check, N*M, M*N))
        rows_check, cols_check, vals_check = findnz(dgdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    else
        @test size(vals) == (size(g)..., size(d)...)
        vals_check = dgdd_check
    end
    @test all(vals .≈ vals_check)
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdd_check[n, m, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    # Check that the partials_dict created by ca2strdict gives the same result as one created using rcdict2strdict.
    partials_dict2 = ca2strdict(J_ca, aviary_input_names, aviary_output_names)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

    return nothing
end

function do_compute_jacvec_product_check_forward(comp, aviary_input_vars=Dict{Symbol,Dict}(), aviary_output_vars=Dict{Symbol,Dict}())
    aviary_input_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_input_vars)
    aviary_output_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_output_vars)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    M, N = size(inputs_dict[get_aviary_input_name(comp, :d)])
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
    inputs_dict[get_aviary_input_name(comp, :b)] .= range(3.0, 4.0; length=N)
    inputs_dict[get_aviary_input_name(comp, :c)] .= range(5.0, 6.0; length=M)
    inputs_dict[get_aviary_input_name(comp, :d)] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)

    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e, f, g = getindex.(Ref(outputs_dict), [get_aviary_output_name(comp, :e), get_aviary_output_name(comp, :f), get_aviary_output_name(comp, :g)])

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the inputs.
    dx = get_dinput_ca(comp)
    dx .= rand(length(dx))
    dinputs_dict = ca2strdict(dx, aviary_input_names)
    doutputs_dict = ca2strdict(get_doutput_ca(comp), aviary_output_names)
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "fwd")

    # Hmm... so how do I check this?
    # Well, just got to do the matrix-vector product myself.
    de_check = similar(e)
    de_check .= 0.0

    # OK, so now I can get a's contribution to the derivative of e.
    # I need to think about deda_check having a size of (N, 1).
    # And is a scalar, and e has size (N,).
    # So then I need to multiply deda_check by dx[:a] and add that to de_check.
    # For deda, the derivative is this:
    deda_check = zeros(N, 1)
    deda_check[:, 1] .= 4*only(a)
    de_check .+= deda_check * dx[:a]

    # Next, the derivative of e wrt b.
    dedb_check = zeros(N, N)
    for n in 1:N
        dedb_check[n, n] = (3*2.1)*b[n]^1.1
    end
    de_check .+= dedb_check * dx[:b]

    # Hmm... will this work?
    # I think I'll flatten things to make sure.
    # Actually don't think that's necessary.
    dedc_check = zeros(N, M)
    for m in 1:M
        for n in 1:N
            dedc_check[n, m] = (4*2.2)*c[m]^1.2
        end
    end
    de_check .+= dedc_check * dx[:c]

    # Hmm... will this work?
    # I think I should reshape things.
    dedd_check = zeros(N, M, N)
    for n in 1:N
        for m in 1:M
            dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
        end
    end
    dxd_rs = reshape(dx[:d], M*N)
    dedd_check_rs = reshape(dedd_check, N, M*N)
    de_check .+= dedd_check_rs * dxd_rs

    # Did all the inputs to `e`, so we're ready to test.
    @test all(doutputs_dict[get_aviary_output_name(comp, :e)] .≈ de_check)

    # Now do `f`.
    df_check = similar(f)
    df_check .= 0.0

    for m in 1:M
        for n in 1:N
            df_check[m, n] += (6*2.4)*only(a)^1.4 * only(dx[:a])
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (7*2.5)*b[n]^1.5 * dx[:b][n]
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (8*2.6)*c[m]^1.6 * dx[:c][m]
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (9*2.7)*d[m, n]^1.7 * dx[:d][m, n]
        end
    end

    @test all(doutputs_dict[get_aviary_output_name(comp, :f)] .≈ df_check)

    # Now do `g`.
    dg_check = similar(g)
    dg_check .= 0.0

    # g is not a function of `a`, so skip that.

    # Derivative wrt b.
    for m in 1:M
        for n in 1:N
            dg_check[n, m] += 10*cos(b[n])*cos(d[m, n]) * dx[:b][n]
        end
    end

    # g is not a function of `c`, so skip that.

    # Derivative wrt d.
    for m in 1:M
        for n in 1:N
            dg_check[n, m] += -10*sin(b[n])*sin(d[m, n]) * dx[:d][m, n]
        end
    end

    @test all(doutputs_dict[get_aviary_output_name(comp, :g)] .≈ dg_check)

    return nothing
end

function do_compute_jacvec_product_check_reverse(comp, aviary_input_vars=Dict{Symbol,Dict}(), aviary_output_vars=Dict{Symbol,Dict}())
    aviary_input_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_input_vars)
    aviary_output_names = Dict{Symbol,String}(k=>get(v, "name", string(k)) for (k, v) in aviary_output_vars)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    M, N = size(inputs_dict[get_aviary_input_name(comp, :d)])
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
    inputs_dict[get_aviary_input_name(comp, :b)] .= range(3.0, 4.0; length=N)
    inputs_dict[get_aviary_input_name(comp, :c)] .= range(5.0, 6.0; length=M)
    inputs_dict[get_aviary_input_name(comp, :d)] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)

    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e, f, g = getindex.(Ref(outputs_dict), [get_aviary_output_name(comp, :e), get_aviary_output_name(comp, :f), get_aviary_output_name(comp, :g)])

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the outputs.
    dy = get_doutput_ca(comp)
    dy .= rand(length(dy))
    doutputs_dict = ca2strdict(dy, aviary_output_names)
    dinputs_dict = ca2strdict(get_dinput_ca(comp), aviary_input_names)
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "rev")

    # Hmm... so how do I check this?
    # Well, just got to do the vector-jacobian product myself.
    da_check = similar(a)
    da_check .= 0.0

    # First, derivative of e wrt a.
    for n in 1:N
        da_check[1] += dy[:e][n] * 4*only(a)
    end

    # Next, derivative of f wrt a.
    for m in 1:M
        for n in 1:N
            da_check[1] += dy[:f][m, n] * (6*2.4)*only(a)^1.4
        end
    end

    # Derivative of g wrt a is zero.

    # Did all the outputs with `a`, so we're ready to test.
    @test all(dinputs_dict[get_aviary_input_name(comp, :a)] .≈ da_check)

    # Next, `b`.
    db_check = similar(b)
    db_check .= 0.0

    # First do the derivative of e wrt b.
    for n in 1:N
        db_check[n] += dy[:e][n] * (3*2.1)*b[n]^1.1
    end

    # Next, derivative of f wrt b.
    for n in 1:N
        for m in 1:M
            db_check[n] += dy[:f][m, n] * (7*2.5)*b[n]^1.5
        end
    end

    # Next, derivative of g wrt b.
    for m in 1:M
        for n in 1:N
            db_check[n] += dy[:g][n, m] * 10*cos(b[n])*cos(d[m, n])
        end
    end

    # That's all the outputs with `b`, so we're ready to check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :b)] .≈ db_check)

    # Now derivatives wrt c.
    dc_check = similar(c)
    dc_check .= 0.0

    # Derivative of `e` wrt c.
    for m in 1:M
        for n in 1:N
            dc_check[m] += dy[:e][n] * (4*2.2)*c[m]^1.2
        end
    end

    # Derivative of `f` wrt c.
    for n in 1:N
        for m in 1:M
            dc_check[m] += dy[:f][m, n] * (8*2.6)*c[m]^1.6
        end
    end

    # Derivative of `g` wrt c is 0.
    
    # Did all the outputs, so ready to check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :c)] .≈ dc_check)

    # Now derivatives wrt d.
    dd_check = similar(d)
    dd_check .= 0.0

    # Derivative of e wrt d.
    for n in 1:N
        for m in 1:M
            dd_check[m, n] += dy[:e][n] * (5*2.3)*d[m, n]^1.3
        end
    end

    # Derivative of f wrt d.
    for n in 1:N
        for m in 1:M
            dd_check[m, n] += dy[:f][m, n] * (9*2.7)*d[m, n]^1.7 
        end
    end

    # Derivative of g wrt d.
    for m in 1:M
        for n in 1:N
            dd_check[m, n] += dy[:g][n, m] * (-10)*sin(b[n])*sin(d[m, n])
        end
    end

    # Now check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :d)] .≈ dd_check)

    return nothing
end
