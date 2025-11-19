struct AutoMatrixFreeTestPrep{TXCA,TYCA,TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    ad_backend::TAD
    disable_prep::Bool
end

function AutoMatrixFreeTestPrep(M, N, ad_type, disable_prep)
    # Also need copies of X_ca and Y_ca.
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

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

    return AutoMatrixFreeTestPrep(M, N, X_ca, Y_ca, ad_backend, disable_prep)
end

struct AutoMatrixFreeShapeByConnTestPrep{TXCA,TYCA,TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    ad_backend::TAD
    disable_prep::Bool
    shape_by_conn_dict::Dict{Symbol,Bool}
end

function AutoMatrixFreeShapeByConnTestPrep(M, N, ad_type, disable_prep)
    N_wrong = 1
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N_wrong), c=zeros(Float64, M), d=zeros(Float64, M, N_wrong))
    Y_ca = ComponentVector(e=zeros(Float64, N_wrong), f=zeros(Float64, M, N_wrong), g=zeros(Float64, N_wrong, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    X_ca[:b] .= 3.0
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N_wrong), M, N_wrong)

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

    shape_by_conn_dict = Dict(:b=>true, :d=>true, :e=>true, :f=>true, :g=>true)

    return AutoMatrixFreeShapeByConnTestPrep(M, N, X_ca, Y_ca, ad_backend, disable_prep, shape_by_conn_dict)
end

function doit_in_place_forward(prep::AutoMatrixFreeTestPrep)
    M = prep.M
    N = prep.N
    ad_backend = prep.ad_backend
    Y_ca = prep.Y_ca
    X_ca = prep.X_ca
    params = (M, N)
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, disable_prep)
    do_compute_check(comp)
    do_compute_jacvec_product_check_forward(comp)
end

for ad in ["forwarddiff", "enzymeforward"]
    prep = AutoMatrixFreeTestPrep(4, 3, ad, false)
    @testset "auto matrix-free, in-place, forward, $ad" verbose=true showtiming=true begin
        doit_in_place_forward(prep)
    end
end

function doit_in_place_forward(prep::AutoMatrixFreeShapeByConnTestPrep)
    ad_backend = prep.ad_backend
    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    disable_prep = prep.disable_prep
    shape_by_conn_dict = prep.shape_by_conn_dict
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; disable_prep, shape_by_conn_dict)

    # Now set the size of b to the correct thing.
    M = prep.M
    N = prep.N
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ range(7.0, 8.0; length=M))
    Y_ca = get_output_ca(comp)
    @test size(Y_ca.e) == (N,)
    @test size(Y_ca.f) == (M, N)
    @test size(Y_ca.g) == (N, M)

    do_compute_check(comp)
    do_compute_jacvec_product_check_forward(comp)
end

for ad in ["forwarddiff", "enzymeforward"]
    prep = AutoMatrixFreeShapeByConnTestPrep(4, 3, ad, false)
    @testset "auto matrix-free, in-place, forward, shape_by_conn, $ad" verbose=true showtiming=true begin
        doit_in_place_forward(prep)
    end
end

function doit_in_place_reverse(prep::AutoMatrixFreeTestPrep)
    M = prep.M
    N = prep.N
    ad_backend = prep.ad_backend
    Y_ca = prep.Y_ca
    X_ca = prep.X_ca
    params = (M, N)
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params, disable_prep)
    do_compute_check(comp)
    do_compute_jacvec_product_check_reverse(comp)
end

for ad in ["reversediff", "enzymereverse"]
    if ad == "reversediff"
        for dp in [true, false]
            prep = AutoMatrixFreeTestPrep(4, 3, ad, dp)
            @testset "auto matrix-free, in-place, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
                doit_in_place_reverse(prep)
            end
        end
    else
        prep = AutoMatrixFreeTestPrep(4, 3, ad, false)
        @testset "auto matrix-free, in-place, reverse, $ad" verbose=true showtiming=true begin
            doit_in_place_reverse(prep)
        end
    end
end

function doit_in_place_reverse(prep::AutoMatrixFreeShapeByConnTestPrep)
    ad_backend = prep.ad_backend
    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    disable_prep = prep.disable_prep
    shape_by_conn_dict = prep.shape_by_conn_dict
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; disable_prep, shape_by_conn_dict)

    # Now set the size of b to the correct thing.
    M = prep.M
    N = prep.N
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ range(7.0, 8.0; length=M))
    Y_ca = get_output_ca(comp)
    @test size(Y_ca.e) == (N,)
    @test size(Y_ca.f) == (M, N)
    @test size(Y_ca.g) == (N, M)

    do_compute_check(comp)
    do_compute_jacvec_product_check_reverse(comp)
end

for ad in ["reversediff", "enzymereverse"]
    if ad == "reversediff"
        for dp in [true, false]
            prep = AutoMatrixFreeShapeByConnTestPrep(4, 3, ad, dp)
            @testset "auto matrix-free, in-place, shape_by_conn, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
                doit_in_place_reverse(prep)
            end
        end
    else
        prep = AutoMatrixFreeShapeByConnTestPrep(4, 3, ad, false)
        @testset "auto matrix-free, in-place, shape_by_conn, reverse, $ad" verbose=true showtiming=true begin
            doit_in_place_reverse(prep)
        end
    end
end

function doit_out_of_place_forward(prep::AutoMatrixFreeTestPrep)
    M = prep.M
    N = prep.N
    ad_backend = prep.ad_backend
    # Y_ca = prep.Y_ca
    X_ca = prep.X_ca
    params = (M, N)
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, disable_prep)
    do_compute_check(comp)
    do_compute_jacvec_product_check_forward(comp)
end

for ad in ["forwarddiff"]
    for dp in [false]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Maybe just because of the out-of-place stuff.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, forward, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_forward(prep)
        end
    end
end

function doit_out_of_place_forward(prep::AutoMatrixFreeShapeByConnTestPrep)
    ad_backend = prep.ad_backend
    X_ca = prep.X_ca
    # Y_ca = prep.Y_ca
    disable_prep = prep.disable_prep
    shape_by_conn_dict = prep.shape_by_conn_dict
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; disable_prep, shape_by_conn_dict)

    # Now set the size of b to the correct thing.
    M = prep.M
    N = prep.N
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ range(7.0, 8.0; length=M))
    Y_ca = get_output_ca(comp)
    @test size(Y_ca.e) == (N,)
    @test size(Y_ca.f) == (M, N)
    @test size(Y_ca.g) == (N, M)

    do_compute_check(comp)
    do_compute_jacvec_product_check_forward(comp)
end

for ad in ["forwarddiff"]
    for dp in [false]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeShapeByConnTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, shape_by_conn, forward, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_forward(prep)
        end
    end
end

function doit_out_of_place_reverse(prep::AutoMatrixFreeTestPrep)
    M = prep.M
    N = prep.N
    ad_backend = prep.ad_backend
    # Y_ca = prep.Y_ca
    X_ca = prep.X_ca
    params = (M, N)
    disable_prep = prep.disable_prep
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, disable_prep)
    do_compute_check(comp)
    do_compute_jacvec_product_check_reverse(comp)
end

for ad in ["reversediff", "zygote"]
    for dp in [false, true]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_reverse(prep)
        end
    end
end

function doit_out_of_place_reverse(prep::AutoMatrixFreeShapeByConnTestPrep)
    M = prep.M
    N = prep.N
    ad_backend = prep.ad_backend
    # Y_ca = prep.Y_ca
    X_ca = prep.X_ca
    params = (M, N)
    disable_prep = prep.disable_prep
    shape_by_conn_dict = prep.shape_by_conn_dict
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, disable_prep, shape_by_conn_dict)

    M = prep.M
    N = prep.N
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    # Make sure the component vectors were set appropriately.
    X_ca = get_input_ca(comp)
    @test X_ca.a ≈ 2.0
    @test size(X_ca.b) == (N,)
    @test all(X_ca.b .≈ 3.0)
    @test size(X_ca.c) == (M,)
    @test all(X_ca.c .≈ range(5.0, 6.0; length=M))
    @test size(X_ca.d) == (M, N)
    # @test all(X_ca.d .≈ 7.0)
    @test all(X_ca.d .≈ range(7.0, 8.0; length=M))
    Y_ca = get_output_ca(comp)
    @test size(Y_ca.e) == (N,)
    @test size(Y_ca.f) == (M, N)
    @test size(Y_ca.g) == (N, M)

    do_compute_check(comp)
    do_compute_jacvec_product_check_reverse(comp)
end

for ad in ["reversediff", "zygote"]
    for dp in [false, true]
        # Disabling prep doesn't appear to work with ForwardDiff.jl, suprisingly.
        # Also Enzyme doesn't like out-of-place callback functions.
        prep = AutoMatrixFreeShapeByConnTestPrep(4, 3, ad, dp)
        @testset "auto matrix-free, out-of-place, shape_by_conn, reverse, $ad, disable_prep=$dp" verbose=true showtiming=true begin
            doit_out_of_place_reverse(prep)
        end
    end
end

