struct AutoMatrixFreeAviaryTestPrep{TAD,TAM,TIV,TOV}
    M::Int
    N::Int
    ad_backend::TAD
    disable_prep::Bool
    aviary_meta_data::TAM
    aviary_input_vars::TIV
    aviary_output_vars::TOV
end

function AutoMatrixFreeAviaryTestPrep(M, N, ad_type, disable_prep)
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
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0
    # aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] = 3.0
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M)
    # aviary_meta_data[aviary_input_names[:d]]["default_value"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    aviary_meta_data[aviary_input_names[:d]]["default_value"] = 7.0

    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoForwardDiff()
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward)
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoReverseDiff()
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoZygote()
    else
        error("unexpected ad_type = $(ad_type)")
    end

    return AutoMatrixFreeAviaryTestPrep(M, N, ad_backend, disable_prep, aviary_meta_data, aviary_input_vars, aviary_output_vars)
end

struct AutoMatrixFreeShapeByConnAviaryTestPrep{TAD,TAM,TIV,TOV}
    M::Int
    N::Int
    ad_backend::TAD
    disable_prep::Bool
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    aviary_meta_data::TAM
    aviary_input_vars::TIV
    aviary_output_vars::TOV
end

function AutoMatrixFreeShapeByConnAviaryTestPrep(M, N, ad_type, disable_prep)
    N_wrong = 1

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

    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoForwardDiff()
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoReverseDiff()
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward)
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoZygote()
    else
        error("unexpected ad_type = $(ad_type)")
    end

    shape_by_conn_dict = Dict(:b=>true, :d=>true, :g=>true)
    copy_shape_dict = Dict(:e=>:b, :f=>:d)

    return AutoMatrixFreeShapeByConnAviaryTestPrep(M, N, ad_backend, disable_prep, shape_by_conn_dict, copy_shape_dict, aviary_meta_data, aviary_input_vars, aviary_output_vars)
end

function doit_in_place_forward(prep::AutoMatrixFreeAviaryTestPrep)
    # `M` and `N` will be passed via the params argument.
    N = prep.N
    M = prep.M
    params = (M, N)

    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_forward(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["forwarddiff", "enzymeforward"]
    prep = AutoMatrixFreeAviaryTestPrep(4, 3, ad, false)
    @testset "auto matrix-free, in-place, forward, Aviary, $ad" verbose=true showtiming=true begin
        doit_in_place_forward(prep)
    end
end

function doit_in_place_forward(prep::AutoMatrixFreeShapeByConnAviaryTestPrep)
    # `M` and `N` will be passed via the params argument.
    N = prep.N
    M = prep.M
    # params = (M, N)

    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; shape_by_conn_dict, copy_shape_dict, disable_prep, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ reshape(range(7.0, 8.0; length=M), M, 1))

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_forward(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["forwarddiff", "enzymeforward"]
    prep = AutoMatrixFreeShapeByConnAviaryTestPrep(4, 3, ad, false)
    @testset "auto matrix-free, in-place, forward, shape_by_conn, Aviary, $ad" verbose=true showtiming=true begin
        doit_in_place_forward(prep)
    end
end

function doit_in_place_reverse(prep::AutoMatrixFreeAviaryTestPrep)
    M = prep.M
    N = prep.N
    params = (M, N)
    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    disable_prep = prep.disable_prep
    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, disable_prep, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_reverse(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["reversediff", "enzymereverse"]
    if ad == "reversediff"
        for dp in [true, false]
            prep = AutoMatrixFreeAviaryTestPrep(4, 3, ad, dp)
            @testset "auto matrix-free, in-place, reverse, Aviary, $ad, disable_prep=$dp" verbose=true showtiming=true begin
                doit_in_place_reverse(prep)
            end
        end
    else
        prep = AutoMatrixFreeAviaryTestPrep(4, 3, ad, false)
        @testset "auto matrix-free, in-place, reverse, Aviary, $ad" verbose=true showtiming=true begin
            doit_in_place_reverse(prep)
        end
    end
end

function doit_in_place_reverse(prep::AutoMatrixFreeShapeByConnAviaryTestPrep)
    # `M` and `N` will be passed via the params argument.
    N = prep.N
    M = prep.M
    # params = (M, N)

    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; disable_prep, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ reshape(range(7.0, 8.0; length=M), M, 1))

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_reverse(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["reversediff", "enzymereverse"]
    if ad == "reversediff"
        for dp in [true, false]
            prep = AutoMatrixFreeShapeByConnAviaryTestPrep(4, 3, ad, dp)
            @testset "auto matrix-free, in-place, reverse, shape_by_conn, Aviary, $ad, disable_prep=$dp" verbose=true showtiming=true begin
                doit_in_place_reverse(prep)
            end
        end
    else
        prep = AutoMatrixFreeShapeByConnAviaryTestPrep(4, 3, ad, false)
        @testset "auto matrix-free, in-place, reverse, shape_by_conn, Aviary, $ad" verbose=true showtiming=true begin
            doit_in_place_reverse(prep)
        end
    end
end

function doit_out_of_place_forward(prep::AutoMatrixFreeAviaryTestPrep)
    # `M` and `N` will be passed via the params argument.
    N = prep.N
    M = prep.M
    params = (M, N)

    X_ca = ComponentVector{Float64}()
    # Y_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_forward(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["forwarddiff"]
    for dp in [false]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Maybe just because of the out-of-place stuff.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeAviaryTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, Aviary, forward, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_forward(prep)
        end
    end
end

function doit_out_of_place_forward(prep::AutoMatrixFreeShapeByConnAviaryTestPrep)
    X_ca = ComponentVector{Float64}()
    # Y_ca = ComponentVector{Float64}()
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; shape_by_conn_dict, copy_shape_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    M = prep.M
    N = prep.N
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

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ reshape(range(7.0, 8.0; length=M), M, 1))

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_forward(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["forwarddiff"]
    for dp in [false]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Maybe just because of the out-of-place stuff.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeShapeByConnAviaryTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, shape_by_conn, Aviary, forward, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_forward(prep)
        end
    end
end

function doit_out_of_place_reverse(prep::AutoMatrixFreeAviaryTestPrep)
    # `M` and `N` will be passed via the params argument.
    N = prep.N
    M = prep.M
    params = (M, N)

    X_ca = ComponentVector{Float64}()
    # Y_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_reverse(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["reversediff", "zygote"]
    for dp in [false, true]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeAviaryTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, Aviary, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_reverse(prep)
        end
    end
end

function doit_out_of_place_reverse(prep::AutoMatrixFreeShapeByConnAviaryTestPrep)
    X_ca = ComponentVector{Float64}()
    # Y_ca = ComponentVector{Float64}()
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data

    ad_backend = prep.ad_backend
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; shape_by_conn_dict, copy_shape_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

    M = prep.M
    N = prep.N
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

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ reshape(range(7.0, 8.0; length=M), M, 1))

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_jacvec_product_check_reverse(comp, aviary_input_vars, aviary_output_vars)
end

for ad in ["reversediff", "zygote"]
    for dp in [false, true]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeShapeByConnAviaryTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, shape_by_conn, Aviary, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_reverse(prep)
        end
    end
end
