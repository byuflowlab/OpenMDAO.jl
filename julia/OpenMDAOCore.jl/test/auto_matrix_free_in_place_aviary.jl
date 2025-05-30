using OpenMDAOCore
using Test
using ADTypes: ADTypes
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore
using Zygote: Zygote

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

function doit_forward(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params = (M, N)

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))
    X_ca = ComponentVector(a=zero(Float64), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N)),
                            # "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zero(Float64)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))
    aviary_input_vars = Dict(:a=>Dict("name"=>"foo:bar:baz:a"),
                             :b=>Dict("name"=>"foo:bar:baz:b"),
                             :c=>Dict("name"=>"foo:bar:baz:c", "shape"=>(M,)),
                             :d=>Dict("name"=>"foo:bar:baz:d"))
    aviary_output_vars = Dict(:e=>Dict("name"=>"foo:bar:baz:e"),
                              :f=>Dict("name"=>"foo:bar:baz:f"),
                              :g=>Dict("name"=>"foo:bar:baz:g"))
    aviary_input_names = Dict(k=>v["name"] for (k, v) in aviary_input_vars)
    aviary_output_names = Dict(k=>v["name"] for (k, v) in aviary_output_vars)

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N) ./ 0.0254^2
    # aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M) .* 2.2369362920544025
    aviary_meta_data[aviary_input_names[:c]]["default_value"] = 5.0 * 2.2369362920544025
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Have the inputs b and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    units_dict = Dict(:b=>"m**2", :c=>"m/s")

    # Now we can create the component.
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoForwardDiff()
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward)
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, disable_prep, units_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    @test get_input_ca(comp).a ≈ 2.0
    @test all(get_input_ca(comp).b .≈ range(3.0, 4.0; length=N))
    # @test all(get_input_ca(comp).c .≈ range(5.0, 6.0; length=M))
    @test size(get_input_ca(comp).c) == (M,)
    @test all(get_input_ca(comp).c .≈ 5.0)
    @test all(get_input_ca(comp).d .≈ reshape(range(7.0, 8.0; length=M*N), M, N))

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
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

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the inputs.
    dx = get_dinput_ca(comp)
    dx .= rand(length(dx))
    dinputs_dict = ca2strdict(dx, aviary_input_names)
    doutputs_dict = ca2strdict(get_doutput_ca(comp), aviary_output_names)
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "fwd")

    # Hmm... so how do I check this?
    # Well, just got to do the matrix-vector product myself.
    de_check = similar(e_check)
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
    df_check = similar(f_check)
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
    dg_check = similar(g_check)
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
end
doit_forward(; disable_prep=false, ad_type="forwarddiff")
doit_forward(; disable_prep=false, ad_type="enzymeforward")

# This fails with 
# in-place: Error During Test at /home/dingraha/.julia/packages/SafeTestsets/raUNr/src/SafeTestsets.jl:30
#   Got exception outside of a @test
#   LoadError: MethodError: no method matching value_and_pushforward(::OpenMDAOCore.var"#32#33"{typeof(Main.var"##in-place#230".f_simple!), Tup
# le{Int64, Int64}}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(e = ViewAxis(1:3, Shaped1DAxis((3,
# ))), f = ViewAxis(4:15, ShapedAxis((4, 3))), g = ViewAxis(16:27, ShapedAxis((3, 4))))}}}, ::DifferentiationInterface.NoPushforwardPrep, ::ADT
# ypes.AutoForwardDiff{nothing, Nothing}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = Vi
# ewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}, ::Tuple{ComponentArrays.C
# omponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1D
# Axis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}})
# doit_forward(; disable_prep=true, ad_type="forwarddiff")

function doit_forward_shape_by_conn(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    # params = (M, N)
    M_wrong = 1

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))
    X_ca = ComponentVector(a=zero(Float64), d=zeros(Float64, M_wrong, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M_wrong, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M_wrong)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M_wrong, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M_wrong, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M_wrong)))
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
    X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N) ./ 0.0254^2
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= 5.0  * 2.2369362920544025
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M_wrong*N), M_wrong, N)

    # Have the inputs b and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    units_dict = Dict(:b=>"m**2", :c=>"m/s")

    # Now we can create the component.
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoForwardDiff()
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward)
    else
        error("unexpected ad_type = $(ad_type)")
    end
    shape_by_conn_dict = Dict(:c=>true, :d=>true, :f=>true, :g=>true)
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; disable_prep, units_dict, shape_by_conn_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    # OK, make sure the inputs and outputs are set to what they should be.
    @test get_input_ca(comp).a ≈ 2.0
    @test all(get_input_ca(comp).b .≈ range(3.0, 4.0; length=N))
    @test all(get_input_ca(comp).c .≈ 5.0)
    @test all(get_input_ca(comp).d .≈ reshape(range(7.0, 8.0; length=M_wrong*N), M_wrong, N))

    @test size(get_output_ca(comp).e) == (N,)
    @test size(get_output_ca(comp).f) == (M, N)
    @test size(get_output_ca(comp).g) == (N, M)

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
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

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the inputs.
    dx = get_dinput_ca(comp)
    dx .= rand(length(dx))
    dinputs_dict = ca2strdict(dx, aviary_input_names)
    doutputs_dict = ca2strdict(get_doutput_ca(comp), aviary_output_names)
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "fwd")

    # Hmm... so how do I check this?
    # Well, just got to do the matrix-vector product myself.
    de_check = similar(e_check)
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
    df_check = similar(f_check)
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
    dg_check = similar(g_check)
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
end
doit_forward_shape_by_conn(; disable_prep=false, ad_type="forwarddiff")
doit_forward_shape_by_conn(; disable_prep=false, ad_type="enzymeforward")

# This fails with 
# in-place: Error During Test at /home/dingraha/.julia/packages/SafeTestsets/raUNr/src/SafeTestsets.jl:30
#   Got exception outside of a @test
#   LoadError: MethodError: no method matching value_and_pushforward(::OpenMDAOCore.var"#32#33"{typeof(Main.var"##in-place#230".f_simple!), Tup
# le{Int64, Int64}}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(e = ViewAxis(1:3, Shaped1DAxis((3,
# ))), f = ViewAxis(4:15, ShapedAxis((4, 3))), g = ViewAxis(16:27, ShapedAxis((3, 4))))}}}, ::DifferentiationInterface.NoPushforwardPrep, ::ADT
# ypes.AutoForwardDiff{nothing, Nothing}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = Vi
# ewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}, ::Tuple{ComponentArrays.C
# omponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1D
# Axis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}})
# doit_forward_shape_by_conn(; disable_prep=true, ad_type="forwarddiff")

function doit_reverse(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params = (M, N)

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    X_ca = ComponentVector(a=zero(Float64), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"Pa", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))

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
    X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N) .* 0.0254^2
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M) .* 2.23694
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Have the inputs b and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    units_dict = Dict(:b=>"m**2", :c=>"m/s")

    # Now we can create the component.
    if ad_type == "reversediff"
        ad_backend = ADTypes.AutoReverseDiff()
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoZygote()
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, disable_prep, units_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
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
doit_reverse(; ad_type="reversediff", disable_prep=false)
doit_reverse(; ad_type="reversediff", disable_prep=true)
doit_reverse(; ad_type="enzymereverse", disable_prep=false)
# doit_reverse(; ad_type="zygote", disable_prep=false)

function doit_reverse_shape_by_conn(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    N_wrong = 1
    # params = (M, N)

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    X_ca = ComponentVector(a=zero(Float64), d=zeros(Float64, M, N_wrong))
    Y_ca = ComponentVector(e=zeros(Float64, N_wrong), f=zeros(Float64, M, N_wrong))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"m", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N_wrong)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M)),
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
    X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N_wrong) .* 0.0254^2
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= 3.0 / 0.0254^2
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M) .* 2.2369362920544025
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N_wrong), M, N_wrong)

    # Have the inputs b and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    units_dict = Dict(:b=>"m**2", :c=>"m/s")

    shape_by_conn_dict = Dict(:b=>true, :d=>true, :e=>true, :f=>true, :g=>true)

    # Now we can create the component.
    if ad_type == "reversediff"
        ad_backend = ADTypes.AutoReverseDiff()
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoZygote()
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; disable_prep, units_dict, shape_by_conn_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    @test get_input_ca(comp).a ≈ 2.0
    @test all(get_input_ca(comp).b .≈ 3.0)
    @test all(get_input_ca(comp).c .≈ range(5.0, 6.0; length=M))
    @test all(get_input_ca(comp).d .≈ reshape(range(7.0, 8.0; length=M*N_wrong), M, N_wrong))

    @test size(get_output_ca(comp).e) == (N,)
    @test size(get_output_ca(comp).f) == (M, N)
    @test size(get_output_ca(comp).g) == (N, M)

    # inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    inputs_dict[get_aviary_input_name(comp, :a)] .= 2.0
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
doit_reverse_shape_by_conn(; ad_type="reversediff", disable_prep=false)
doit_reverse_shape_by_conn(; ad_type="reversediff", disable_prep=true)
doit_reverse_shape_by_conn(; ad_type="enzymereverse", disable_prep=false)
# doit_reverse(; ad_type="zygote", disable_prep=false)
