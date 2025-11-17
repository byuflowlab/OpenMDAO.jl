using OpenMDAOCore
using Test
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using SparseArrays: sparse, findnz, nnz, issparse
using SparseMatrixColorings: SparseMatrixColorings
using ADTypes: ADTypes
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore
using Zygote: Zygote

struct AutosparseManualTestPrep{TXCA, TYCA, TJCA, TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    J_ca::TJCA
    ad_backend::TAD
end

function AutosparseManualTestPrep(M, N, ad_type)
    # Also need copies of X_ca and Y_ca.
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

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

    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(sparse(getdata(J_ca))), coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    elseif ad_type == "reversediff"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoReverseDiff(); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(sparse(getdata(J_ca))), coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(sparse(getdata(J_ca))), coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(sparse(getdata(J_ca))), coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoSparse(ADTypes.AutoZygote(); sparsity_detector=ADTypes.KnownJacobianSparsityDetector(sparse(getdata(J_ca))), coloring_algorithm=SparseMatrixColorings.GreedyColoringAlgorithm())
    else
        error("unexpected ad_type = $(ad_type)")
    end

    return AutosparseManualTestPrep(M, N, X_ca, Y_ca, J_ca, ad_backend)
end

function doit_in_place(prep::AutosparseManualTestPrep)
    # `M` and `N` will be passed via the params argument.
    M = prep.M
    N = prep.N
    params_simple = (M, N)

    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    ad_backend = prep.ad_backend

    # Now we can create the component.
    comp = SparseADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params=params_simple)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
    p = AutosparseManualTestPrep(4, 3, ad)
    @testset "autosparse_manual, in-place, $ad" verbose=true showtiming=true begin
        doit_in_place(p)
    end
end

function doit_out_of_place(prep::AutosparseManualTestPrep)
    params_simple = nothing

    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    J_ca = prep.J_ca
    ad_backend = prep.ad_backend

    # Now we can create the component.
    comp = SparseADExplicitComp(ad_backend, f_simple, X_ca; params=params_simple)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for ad in ["forwarddiff", "reversediff", "zygote"]
    p = AutosparseManualTestPrep(4, 3, ad)
    @testset "autosparse_manual, out-of-place, $ad" verbose=true showtiming=true begin
        doit_out_of_place(p)
    end
end
