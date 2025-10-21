using OpenMDAOCore
using Test
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using SparseArrays: sparse, findnz, nnz, issparse
using ADTypes: ADTypes
using SparseMatrixColorings: SparseMatrixColorings
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote
using EnzymeCore: EnzymeCore
using Enzyme: Enzyme

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

function doit_in_place(; sparse_detect_method, ad_type)
    # println("autosparse automatic in-place test with ad_type = $(ad_type), sparse_detect_method = $(sparse_detect_method)")
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params = (M, N)

    # Also need copies of X_ca and Y_ca.
    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            # "foo:bar:baz:b"=>Dict("units"=>"m**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:b"=>Dict("units"=>"m**2", "default_value"=>zero(Float64)),
                            "foo:bar:baz:c"=>Dict("units"=>"m/s", "default_value"=>zeros(Float64, M)),
                            # "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zero(Float64)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))
    aviary_input_vars = Dict(:a=>Dict("name"=>"foo:bar:baz:a"),
                             :b=>Dict("name"=>"foo:bar:baz:b", "shape"=>N),
                             :c=>Dict("name"=>"foo:bar:baz:c"),
                             :d=>Dict("name"=>"foo:bar:baz:d", "shape"=>(M, N)))
    aviary_output_vars = Dict(:e=>Dict("name"=>"foo:bar:baz:e"),
                              :f=>Dict("name"=>"foo:bar:baz:f"),
                              :g=>Dict("name"=>"foo:bar:baz:g"))
    aviary_input_names = Dict(k=>v["name"] for (k, v) in aviary_input_vars)
    aviary_output_names = Dict(k=>v["name"] for (k, v) in aviary_output_vars)

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    # X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] = 3.0
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M)
    # aviary_meta_data[aviary_input_names[:d]]["default_value"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:d]]["default_value"] = 7.0

    # Now we can create the component.
    sparse_atol = 1e-10
    sparsity_detector = PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=sparse_detect_method)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoReverseDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse); sparsity_detector, coloring_algorithm)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoZygote(); sparsity_detector, coloring_algorithm)
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = SparseADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    @test all(X_ca.d .≈ 7.0)

    # Now run all the checks from the previous case.
    rcdict = get_rows_cols_dict(comp)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
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

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    J_ca_sparse = get_jacobian_ca(comp)
    @test issparse(getdata(J_ca_sparse))
    @test size(getdata(J_ca_sparse)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
    @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    partials_dict = rcdict2strdict(rcdict; cnames=aviary_input_names, rnames=aviary_output_names)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    # Complex step size.
    h = 1e-10

    rows, cols = rcdict[:e, :a]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ deda_check[n]
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:e, :b]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedb_check[n, n]
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:e, :c]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedc_check[n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:e, :d]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedd_check[n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:f, :a]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        for m in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfda_check[m, n]
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:f, :b]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdb_check[m, n, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:f, :c]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdc_check[m, n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:f, :d]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdd_check[m, n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:g, :a]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :a)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for m in 1:M
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:g, :b]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdb_check[n, m, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end


    rows, cols = rcdict[:g, :c]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :c)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:g, :d]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :d)]
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
    partials_dict2 = ca2strdict(J_ca_sparse, aviary_input_names, aviary_output_names)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        println("autosparse_automatic_aviary, in-place, $ad, $sdm")
        @time doit_in_place(; sparse_detect_method=sdm, ad_type=ad)
    end
   
    # I don't think zygote works with in-place callback functions.
    # doit_in_place(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_in_place_shape_by_conn(; sparse_detect_method, ad_type)
    # println("autosparse automatic in-place test with ad_type = $(ad_type), sparse_detect_method = $(sparse_detect_method)")
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    # params = (M, N)

    # Also need copies of X_ca and Y_ca.
    N_wrong = 1
    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"m**2", "default_value"=>zeros(Float64, N_wrong)),
                            "foo:bar:baz:c"=>Dict("units"=>"m/s", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N_wrong)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N_wrong)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N_wrong)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N_wrong, M)))
    aviary_input_vars = Dict(:a=>Dict("name"=>"foo:bar:baz:a"),
                             :b=>Dict("name"=>"foo:bar:baz:b"),
                             :c=>Dict("name"=>"foo:bar:baz:c"),
                             :d=>Dict("name"=>"foo:bar:baz:d"))
    aviary_output_vars = Dict(:e=>Dict("name"=>"foo:bar:baz:e"),
                              :f=>Dict("name"=>"foo:bar:baz:f"),
                              :g=>Dict("name"=>"foo:bar:baz:g"))
    aviary_input_names = Dict(k=>v["name"] for (k, v) in aviary_input_vars)
    aviary_output_names = Dict(k=>v["name"] for (k, v) in aviary_output_vars)

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    # X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N_wrong)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= 3.0
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:d]]["default_value"] .= reshape(range(7.0, 8.0; length=M*N_wrong), M, N_wrong)

    # Now we can create the component.
    sparse_atol = 1e-10
    sparsity_detector = PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=sparse_detect_method)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoReverseDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse); sparsity_detector, coloring_algorithm)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoZygote(); sparsity_detector, coloring_algorithm)
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = SparseADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; shape_by_conn_dict=Dict(:b=>true, :d=>true, :e=>true, :f=>true, :g=>true), aviary_input_vars, aviary_output_vars, aviary_meta_data)

    # Now set the size of b to the correct thing.
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # Now run all the checks from the previous case.
    rcdict = get_rows_cols_dict(comp)

    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
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

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    J_ca_sparse = get_jacobian_ca(comp)
    @test issparse(getdata(J_ca_sparse))
    @test size(getdata(J_ca_sparse)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
    @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    partials_dict = rcdict2strdict(rcdict; cnames=aviary_input_names, rnames=aviary_output_names)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    # Complex step size.
    h = 1e-10

    rows, cols = rcdict[:e, :a]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ deda_check[n]
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:e, :b]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedb_check[n, n]
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:e, :c]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedc_check[n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:e, :d]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedd_check[n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:f, :a]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        for m in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfda_check[m, n]
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:f, :b]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdb_check[m, n, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:f, :c]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdc_check[m, n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:f, :d]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdd_check[m, n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:g, :a]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :a)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for m in 1:M
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:g, :b]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdb_check[n, m, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end


    rows, cols = rcdict[:g, :c]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :c)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:g, :d]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :d)]
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
    partials_dict2 = ca2strdict(J_ca_sparse, aviary_input_names, aviary_output_names)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        println("autosparse_automatic_aviary, in-place shape by conn, $ad, $sdm")
        @time doit_in_place_shape_by_conn(; sparse_detect_method=sdm, ad_type=ad)
    end
   
    # I don't think zygote works with in-place callback functions.
    # doit_in_place_shape_by_conn(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_out_of_place(; ad_type, sparse_detect_method)
    # println("autosparse automatic out-of-place test with ad_type = $(ad_type), sparse_detect_method = $(sparse_detect_method)")
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    # params = (M, N)
    params = nothing

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # X_ca = ComponentVector{Float64}()
    X_ca = ComponentVector(b=zeros(Float64, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"m**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"m/s", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))
    aviary_input_vars = Dict(:a=>Dict("name"=>"foo:bar:baz:a"),
                             # :b=>Dict("name"=>"foo:bar:baz:b"),
                             :c=>Dict("name"=>"foo:bar:baz:c"),
                             :d=>Dict("name"=>"foo:bar:baz:d"))
    aviary_output_vars = Dict(:e=>Dict("name"=>"foo:bar:baz:e"),
                              :f=>Dict("name"=>"foo:bar:baz:f"),
                              :g=>Dict("name"=>"foo:bar:baz:g"))
    aviary_input_names = Dict(k=>v["name"] for (k, v) in aviary_input_vars)
    aviary_output_names = Dict(k=>v["name"] for (k, v) in aviary_output_vars)

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    # X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N)
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:d]]["default_value"] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Now we can create the component.
    sparse_atol = 1e-10
    sparsity_detector = PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=sparse_detect_method)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoReverseDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse); sparsity_detector, coloring_algorithm)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoZygote(); sparsity_detector, coloring_algorithm)
    else
        error("unexpected ad_type = $(ad_type)")
    end

    units_dict = Dict(:b=>"m**2")

    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; params, units_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # Now run all the checks from the previous case.
    rcdict = get_rows_cols_dict(comp)

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    # inputs_dict["a"] .= 2.0
    # inputs_dict["b"] .= range(3.0, 4.0; length=N)
    # inputs_dict["c"] .= range(5.0, 6.0; length=M)
    # inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    # outputs_dict = ca2strdict(get_output_ca(comp))
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)
    # outputs_dict = Dict(oname_aviary=>aviary_meta_data[oname_aviary]["default_value"] for (oname, oname_aviary) in aviary_output_names)

    inputs_dict_cs = ca2strdict(get_input_ca(ComplexF64, comp), aviary_input_names)
    # inputs_dict_cs["a"] .= inputs_dict["a"]
    # inputs_dict_cs["b"] .= inputs_dict["b"]
    # inputs_dict_cs["c"] .= inputs_dict["c"]
    # inputs_dict_cs["d"] .= inputs_dict["d"]
    inputs_dict_cs[get_aviary_input_name(comp, :a)] .= inputs_dict[get_aviary_input_name(comp, :a)]
    inputs_dict_cs[get_aviary_input_name(comp, :b)] .= inputs_dict[get_aviary_input_name(comp, :b)]
    inputs_dict_cs[get_aviary_input_name(comp, :c)] .= inputs_dict[get_aviary_input_name(comp, :c)]
    inputs_dict_cs[get_aviary_input_name(comp, :d)] .= inputs_dict[get_aviary_input_name(comp, :d)]
    # outputs_dict_cs = ca2strdict(similar(Y_ca, ComplexF64))
    # outputs_dict_cs = ca2strdict(get_output_ca(ComplexF64, comp))
    # outputs_dict_cs = Dict(k=>ComplexF64.(v) for (k, v) in outputs_dict)
    outputs_dict_cs = ca2strdict(get_output_ca(ComplexF64, comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    # a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    J_ca_sparse = get_jacobian_ca(comp)
    @test issparse(getdata(J_ca_sparse))
    @test size(getdata(J_ca_sparse)) == (length(get_callback(comp)(get_input_ca(comp))), length(get_input_ca(comp)))
    @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    # partials_dict = rcdict2strdict(rcdict)
    partials_dict = rcdict2strdict(rcdict; cnames=aviary_input_names, rnames=aviary_output_names)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    # Complex step size.
    h = 1e-10

    rows, cols = rcdict[:e, :a]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ deda_check[n]
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:e, :b]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedb_check[n, n]
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:e, :c]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedc_check[n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:e, :d]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedd_check[n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:f, :a]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        for m in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfda_check[m, n]
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:f, :b]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdb_check[m, n, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:f, :c]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdc_check[m, n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:f, :d]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdd_check[m, n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:g, :a]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :a)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for m in 1:M
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:g, :b]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdb_check[n, m, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:g, :c]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :c)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:g, :d]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :d)]
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
    partials_dict2 = ca2strdict(J_ca_sparse, aviary_input_names, aviary_output_names)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        println("autosparse_automatic_aviary, out-of-place, $ad, $sdm")
        @time doit_out_of_place(; sparse_detect_method=sdm, ad_type=ad)
    end

    # Got exception outside of a @test
    # LoadError: UndefRefError: access to undefined reference
    # Stacktrace:
    #   [1] LLVM.Value(ref::Ptr{LLVM.API.LLVMOpaqueValue})
    #     @ LLVM ~/.julia/packages/LLVM/b3kFs/src/core/value.jl:39
    #   [2] jl_nthfield_fwd
    #     @ ~/.julia/packages/Enzyme/QsaeA/src/rules/typeunstablerules.jl:1554 [inlined]
    #   [3] jl_nthfield_fwd_cfunc(B::Ptr{LLVM.API.LLVMOpaqueBuilder}, OrigCI::Ptr{LLVM.API.LLVMOpaqueValue}, gutils::Ptr{Nothing}, normalR::Ptr{Ptr{LLVM.API.LLVMOpaqueValue}}, shadowR::Ptr{Ptr{LLVM.API.LLVMOpaqueVal
#   ue}})
    #     @ Enzyme.Compiler ~/.julia/packages/Enzyme/QsaeA/src/rules/llvmrules.jl:75
    # doit_out_of_place(; sparse_detect_method=sdm, ad_type="enzymeforward")

    # Giant scary stacktrace from this one:
    # doit_out_of_place(; sparse_detect_method=sdm, ad_type="enzymereverse")

end

function doit_out_of_place_shape_by_conn(; ad_type, sparse_detect_method)
    # println("autosparse automatic out-of-place test with ad_type = $(ad_type), sparse_detect_method = $(sparse_detect_method)")
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    M_wrong = 1
    # params = (M, N)
    params = nothing

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # X_ca = ComponentVector{Float64}()
    X_ca = ComponentVector(b=zeros(Float64, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"m**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"m/s", "default_value"=>zeros(Float64, M_wrong)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M_wrong, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M_wrong, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M_wrong)))
    aviary_input_vars = Dict(:a=>Dict("name"=>"foo:bar:baz:a"),
                             # :b=>Dict("name"=>"foo:bar:baz:b"),
                             :c=>Dict("name"=>"foo:bar:baz:c"),
                             :d=>Dict("name"=>"foo:bar:baz:d"))
    aviary_output_vars = Dict(:e=>Dict("name"=>"foo:bar:baz:e"),
                              :f=>Dict("name"=>"foo:bar:baz:f"),
                              :g=>Dict("name"=>"foo:bar:baz:g"))
    aviary_input_names = Dict(k=>v["name"] for (k, v) in aviary_input_vars)
    aviary_output_names = Dict(k=>v["name"] for (k, v) in aviary_output_vars)

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    # X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N)
    # aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= 5.0
    aviary_meta_data[aviary_input_names[:d]]["default_value"] .= reshape(range(7.0, 8.0; length=M_wrong*N), M_wrong, N)

    # Now we can create the component.
    sparse_atol = 1e-10
    sparsity_detector = PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=sparse_detect_method)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoReverseDiff(); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward); sparsity_detector, coloring_algorithm)
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse); sparsity_detector, coloring_algorithm)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoZygote(); sparsity_detector, coloring_algorithm)
    else
        error("unexpected ad_type = $(ad_type)")
    end

    units_dict = Dict(:b=>"m**2")

    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; params, units_dict, shape_by_conn_dict=Dict(:c=>true, :d=>true, :f=>true, :g=>true), aviary_input_vars, aviary_output_vars, aviary_meta_data)

    # Now set the size of b to the correct thing.
    input_sizes = Dict(:c=>M, :d=>(M, N))
    output_sizes = Dict(:f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # Also make sure the initial values of everything are what we expect.
    @test get_input_ca(comp).a ≈ 2.0
    @test all(get_input_ca(comp).b .≈ range(3.0, 4.0; length=N))
    @test all(get_input_ca(comp).c .≈ 5.0)
    # So, the default value for d is a matrix with size (1, N), values of range(7.0, 8.0, length=N)
    # Then that will be broadcasted to the new shape of (M, N).
    # So I should be able to do the same broadcasting, I guess.
    @test all(get_input_ca(comp).d .≈ reshape(range(7.0, 8.0; length=N), 1, N))

    # For out-of-place explicit callback functions there is no stored output component vector, so the default values don't matter.
    # But I can at least test that they're the correct shape.
    @test size(get_output_ca(comp).e) == (N,)
    @test size(get_output_ca(comp).f) == (M, N)
    @test size(get_output_ca(comp).g) == (N, M)

    # Now run all the checks from the previous case.
    rcdict = get_rows_cols_dict(comp)

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    # inputs_dict["a"] .= 2.0
    # inputs_dict["b"] .= range(3.0, 4.0; length=N)
    # inputs_dict["c"] .= range(5.0, 6.0; length=M)
    # inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    # outputs_dict = ca2strdict(get_output_ca(comp))
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)
    # outputs_dict = Dict(oname_aviary=>aviary_meta_data[oname_aviary]["default_value"] for (oname, oname_aviary) in aviary_output_names)

    inputs_dict_cs = ca2strdict(get_input_ca(ComplexF64, comp), aviary_input_names)
    # inputs_dict_cs["a"] .= inputs_dict["a"]
    # inputs_dict_cs["b"] .= inputs_dict["b"]
    # inputs_dict_cs["c"] .= inputs_dict["c"]
    # inputs_dict_cs["d"] .= inputs_dict["d"]
    inputs_dict_cs[get_aviary_input_name(comp, :a)] .= inputs_dict[get_aviary_input_name(comp, :a)]
    inputs_dict_cs[get_aviary_input_name(comp, :b)] .= inputs_dict[get_aviary_input_name(comp, :b)]
    inputs_dict_cs[get_aviary_input_name(comp, :c)] .= inputs_dict[get_aviary_input_name(comp, :c)]
    inputs_dict_cs[get_aviary_input_name(comp, :d)] .= inputs_dict[get_aviary_input_name(comp, :d)]
    # outputs_dict_cs = ca2strdict(similar(Y_ca, ComplexF64))
    # outputs_dict_cs = ca2strdict(get_output_ca(ComplexF64, comp))
    # outputs_dict_cs = Dict(k=>ComplexF64.(v) for (k, v) in outputs_dict)
    outputs_dict_cs = ca2strdict(get_output_ca(ComplexF64, comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    # a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    J_ca_sparse = get_jacobian_ca(comp)
    @test issparse(getdata(J_ca_sparse))
    @test size(getdata(J_ca_sparse)) == (length(get_callback(comp)(get_input_ca(comp))), length(get_input_ca(comp)))
    @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
    # partials_dict = rcdict2strdict(rcdict)
    partials_dict = rcdict2strdict(rcdict; cnames=aviary_input_names, rnames=aviary_output_names)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

    # Complex step size.
    h = 1e-10

    rows, cols = rcdict[:e, :a]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ deda_check[n]
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:e, :b]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedb_check[n, n]
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:e, :c]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedc_check[n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:e, :d]
    vals = partials_dict[get_aviary_output_name(comp, :e), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :e)][n])/h ≈ dedd_check[n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:f, :a]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :a)]
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
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for n in 1:N
        for m in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfda_check[m, n]
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:f, :b]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdb_check[m, n, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:f, :c]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :c)]
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
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdc_check[m, n, m]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:f, :d]
    vals = partials_dict[get_aviary_output_name(comp, :f), get_aviary_input_name(comp, :d)]
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
    # Check with complex step.
    for n in 1:N
        for m in 1:M
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n] + im*h
            OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :f)][m, n])/h ≈ dfdd_check[m, n, m, n]
            inputs_dict_cs[get_aviary_input_name(comp, :d)][m, n] = inputs_dict[get_aviary_input_name(comp, :d)][m, n]
        end
    end

    rows, cols = rcdict[:g, :a]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :a)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1] + im*h
    OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
    for m in 1:M
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
    end
    inputs_dict_cs[get_aviary_input_name(comp, :a)][1] = inputs_dict[get_aviary_input_name(comp, :a)][1]

    rows, cols = rcdict[:g, :b]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :b)]
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
    # Check with complex step.
    for n in 1:N
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for m in 1:M
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ dgdb_check[n, m, n]
        end
        inputs_dict_cs[get_aviary_input_name(comp, :b)][n] = inputs_dict[get_aviary_input_name(comp, :b)][n]
    end

    rows, cols = rcdict[:g, :c]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :c)]
    @test size(vals) == (0,)
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
    @test eltype(vals) == Float64
    # Check with complex step.
    for m in 1:M
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m] + im*h
        OpenMDAOCore.compute!(comp, inputs_dict_cs, outputs_dict_cs)
        for n in 1:N
            @test imag(outputs_dict_cs[get_aviary_output_name(comp, :g)][n, m])/h ≈ 0
        end
        inputs_dict_cs[get_aviary_input_name(comp, :c)][m] = inputs_dict[get_aviary_input_name(comp, :c)][m]
    end

    rows, cols = rcdict[:g, :d]
    vals = partials_dict[get_aviary_output_name(comp, :g), get_aviary_input_name(comp, :d)]
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
    partials_dict2 = ca2strdict(J_ca_sparse, aviary_input_names, aviary_output_names)
    # So I think partitals_dict has sparse arrays, but partials_dict2 might just have dense arrays.
    # Ah, no, partials_dict has just plain vectors, but partials_dict2 has reshaped sparse arrays.
    @test keys(partials_dict2) == keys(partials_dict)
    OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict2)
    for k in keys(partials_dict2)
        @test all(OpenMDAOCore._maybe_nonzeros(partials_dict2[k]) .≈ OpenMDAOCore._maybe_nonzeros(partials_dict[k]))
    end

end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        println("autosparse_automatic_aviary, out-of-place shape by conn, $ad, $sdm")
        @time doit_out_of_place_shape_by_conn(; sparse_detect_method=sdm, ad_type=ad)
    end

    # Got exception outside of a @test
    # LoadError: UndefRefError: access to undefined reference
    # Stacktrace:
    #   [1] LLVM.Value(ref::Ptr{LLVM.API.LLVMOpaqueValue})
    #     @ LLVM ~/.julia/packages/LLVM/b3kFs/src/core/value.jl:39
    #   [2] jl_nthfield_fwd
    #     @ ~/.julia/packages/Enzyme/QsaeA/src/rules/typeunstablerules.jl:1554 [inlined]
    #   [3] jl_nthfield_fwd_cfunc(B::Ptr{LLVM.API.LLVMOpaqueBuilder}, OrigCI::Ptr{LLVM.API.LLVMOpaqueValue}, gutils::Ptr{Nothing}, normalR::Ptr{Ptr{LLVM.API.LLVMOpaqueValue}}, shadowR::Ptr{Ptr{LLVM.API.LLVMOpaqueVal
#   ue}})
    #     @ Enzyme.Compiler ~/.julia/packages/Enzyme/QsaeA/src/rules/llvmrules.jl:75
    # doit_out_of_place_shape_by_conn(; sparse_detect_method=sdm, ad_type="enzymeforward")

    # Giant scary stacktrace from this one:
    # doit_out_of_place_shape_by_conn(; sparse_detect_method=sdm, ad_type="enzymereverse")
end
