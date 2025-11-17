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

struct AutosparseAutomaticTestPrep{TXCA, TYCA, TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    ad_backend::TAD
end

function AutosparseAutomaticTestPrep(M, N, ad_type, sparse_detect_method)
    # Also need copies of X_ca and Y_ca.
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

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

    return AutosparseAutomaticTestPrep(M, N, X_ca, Y_ca, ad_backend)
end

struct AutosparseAutomaticShapeByConnTestPrep{TXCA, TYCA, TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    ad_backend::TAD
    shape_by_conn_dict::Dict{Symbol,Bool}
    copy_shape_dict::Dict{Symbol,Symbol}
end

function AutosparseAutomaticShapeByConnTestPrep(M, N, ad_type, sparse_detect_method)
    # Also need copies of X_ca and Y_ca.
    N_wrong = 1
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N_wrong), c=zeros(Float64, M), d=zeros(Float64, M, N_wrong))
    Y_ca = ComponentVector(e=zeros(Float64, N_wrong), f=zeros(Float64, M, N_wrong), g=zeros(Float64, N_wrong, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:b] .= 3.0
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N_wrong), M, N_wrong)

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

    return AutosparseAutomaticShapeByConnTestPrep(M, N, X_ca, Y_ca, ad_backend, shape_by_conn_dict, copy_shape_dict)
end

function doit_in_place(prep::AutosparseAutomaticTestPrep)
    # `M` and `N` will be passed via the params argument.
    M = prep.M
    N = prep.N
    params = (M, N)

    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    ad_backend = prep.ad_backend
    comp = SparseADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params=params)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        p = AutosparseAutomaticTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, in-place, $ad, $sdm" verbose=true showtiming=true begin
            doit_in_place(p)
        end
    end
   
    # I don't think zygote works with in-place callback functions.
    # doit_in_place(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_in_place(prep::AutosparseAutomaticShapeByConnTestPrep)
    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    ad_backend = prep.ad_backend
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict

    # Now we can create the component.
    comp = SparseADExplicitComp(ad_backend, f_simple_no_params!, Y_ca, X_ca; shape_by_conn_dict, copy_shape_dict)

    # Now set the size of b to the correct thing.
    N = prep.N
    M = prep.M
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for sdm in [:direct, :iterative]
    # for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
    #     println("autosparse_automatic, in-place shape by conn, $ad, $sdm")
    #     @time doit_in_place_shape_by_conn(; sparse_detect_method=sdm, ad_type=ad)
    # end
    for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
        p = AutosparseAutomaticShapeByConnTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, in-place, shape_by_conn, $ad, $sdm" verbose=true showtiming=true begin
            doit_in_place(p)
        end
    end

    # I don't think zygote works with in-place callback functions.
    # doit_in_place(; sparse_detect_method=sdm, ad_type="zygote")
end

function doit_out_of_place(prep::AutosparseAutomaticTestPrep)
    X_ca = prep.X_ca
    ad_backend = prep.ad_backend
    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca)

      do_compute_check(comp)
      do_compute_partials_check(comp)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        p = AutosparseAutomaticTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, out-of-place, $ad, $sdm" verbose=true showtiming=true begin
            doit_out_of_place(p)
        end
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

function doit_out_of_place(prep::AutosparseAutomaticShapeByConnTestPrep)
    X_ca = prep.X_ca
    # Y_ca = prep.Y_ca
    ad_backend = prep.ad_backend
    shape_by_conn_dict = prep.shape_by_conn_dict
    copy_shape_dict = prep.copy_shape_dict
    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; shape_by_conn_dict, copy_shape_dict)

    M = prep.M
    N = prep.N
    input_sizes = Dict(:b=>N, :d=>(M, N))
    output_sizes = Dict(:e=>N, :f=>(M, N), :g=>(N, M))
    comp = OpenMDAOCore.update_prep(comp, input_sizes, output_sizes)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for sdm in [:direct, :iterative]
    for ad in ["forwarddiff", "reversediff", "zygote"]
        p = AutosparseAutomaticShapeByConnTestPrep(4, 3, ad, sdm)
        @testset "autosparse_automatic, out-of-place, shape_by_conn, $ad, $sdm" verbose=true showtiming=true begin
            doit_out_of_place(p)
        end
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
