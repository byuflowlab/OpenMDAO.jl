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

struct AutosparseAutomaticAviaryTestPrep{TAD,TAM,TIV,TOV}
    M::Int
    N::Int
    ad_backend::TAD
    aviary_meta_data::TAM
    aviary_input_vars::TIV
    aviary_output_vars::TOV
end

function AutosparseAutomaticAviaryTestPrep(M, N, ad_type, sparse_detect_method)
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

    return AutosparseAutomaticAviaryTestPrep(M, N, ad_backend, aviary_meta_data, aviary_input_vars, aviary_output_vars)
end

struct AutosparseAutomaticShapeByConnAviaryTestPrep{TAD,TAM,TIV,TOV}
    M::Int
    N::Int
    ad_backend::TAD
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
    aviary_meta_data::TAM
    aviary_input_vars::TIV
    aviary_output_vars::TOV
end

function AutosparseAutomaticShapeByConnAviaryTestPrep(M, N, ad_type, sparse_detect_method)
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

    shape_by_conn_dict = Dict(:b=>true, :d=>true, :g=>true)
    copy_shape_dict = Dict(:e=>:b, :f=>:d)

    return AutosparseAutomaticShapeByConnAviaryTestPrep(M, N, ad_backend, shape_by_conn_dict, copy_shape_dict, aviary_meta_data, aviary_input_vars, aviary_output_vars)
end

function doit_in_place(prep::AutosparseAutomaticAviaryTestPrep)
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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_partials_check(comp, aviary_input_vars, aviary_output_vars)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        p = AutosparseAutomaticAviaryTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, in-place, Aviary, $ad, $sdm" verbose=true showtiming=true begin
            doit_in_place(p)
        end
    end
   
    # I don't think zygote works with in-place callback functions.
    # doit_in_place(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_in_place(prep::AutosparseAutomaticShapeByConnAviaryTestPrep)
    N = prep.N
    M = prep.M

    # Also need copies of X_ca and Y_ca.
    # N_wrong = 1
    X_ca = ComponentVector{Float64}()
    Y_ca = ComponentVector{Float64}()
    ad_backend = prep.ad_backend
    shape_by_conn_dict = prep.shape_by_conn_dict
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data
    comp = SparseADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; shape_by_conn_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_partials_check(comp, aviary_input_vars, aviary_output_vars)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        p = AutosparseAutomaticShapeByConnAviaryTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, in-place, shape_by_conn, aviary, $ad, $sdm" verbose=true showtiming=true begin
            doit_in_place(p)
        end
    end
   
    # I don't think zygote works with in-place callback functions.
    # doit_in_place_shape_by_conn(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_out_of_place(prep::AutosparseAutomaticAviaryTestPrep)
    X_ca = ComponentVector{Float64}()
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data
    ad_backend = prep.ad_backend
    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; aviary_input_vars, aviary_output_vars, aviary_meta_data)

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
    N = prep.N
    M = prep.M
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    @test all(X_ca.d .≈ 7.0)

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_partials_check(comp, aviary_input_vars, aviary_output_vars)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        p = AutosparseAutomaticAviaryTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, out-of-place, Aviary, $ad, $sdm" verbose=true showtiming=true begin
            doit_out_of_place(p)
        end
    end
end

function doit_out_of_place(prep::AutosparseAutomaticShapeByConnAviaryTestPrep)
    X_ca = ComponentVector{Float64}()
    ad_backend = prep.ad_backend
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict
    aviary_input_vars = prep.aviary_input_vars
    aviary_output_vars = prep.aviary_output_vars
    aviary_meta_data = prep.aviary_meta_data
    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; shape_by_conn_dict, copy_shape_dict, aviary_input_vars, aviary_output_vars, aviary_meta_data)

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

    do_compute_check(comp, aviary_input_vars, aviary_output_vars)
    do_compute_partials_check(comp, aviary_input_vars, aviary_output_vars)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        p = AutosparseAutomaticShapeByConnAviaryTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, out-of-place, shape_by_conn, Aviary, $ad, $sdm" verbose=true showtiming=true begin
            doit_out_of_place(p)
        end
    end
end
