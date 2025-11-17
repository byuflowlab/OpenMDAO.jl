using OpenMDAOCore
using Test
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using ADTypes: ADTypes
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore
using Zygote: Zygote

struct AutoDenseTestPrep{TXCA, TYCA, TAD}
    M::Int
    N::Int
    X_ca::TXCA
    Y_ca::TYCA
    ad_backend::TAD
end

function AutoDenseTestPrep(M, N, ad_type)
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

    return AutoDenseTestPrep(M, N, X_ca, Y_ca, ad_backend)
end

function doit_in_place(prep::AutoDenseTestPrep)
    # `M` and `N` will be passed via the params argument.
    M = prep.M
    N = prep.N
    params_simple = (M, N)

    X_ca = prep.X_ca
    Y_ca = prep.Y_ca
    ad_backend = prep.ad_backend

    # Now we can create the component.
    comp = DenseADExplicitComp(ad_backend, f_simple!, Y_ca, X_ca; params=params_simple)

    # Do the checks.
    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for ad in ["forwarddiff", "reversediff", "enzymeforward", "enzymereverse"]
    p = AutoDenseTestPrep(4, 3, ad)
    @testset "autodense, in-place, $ad" verbose=true showtiming=true begin
        doit_in_place(p)
    end
end

function doit_out_of_place(prep::AutoDenseTestPrep)
    params_simple = nothing

    X_ca = prep.X_ca
    ad_backend = prep.ad_backend

    # Now we can create the component.
    comp = DenseADExplicitComp(ad_backend, f_simple, X_ca; params=params_simple)

    do_compute_check(comp)
    do_compute_partials_check(comp)
end

for ad in ["forwarddiff", "reversediff", "zygote"]
    p = AutoDenseTestPrep(4, 3, ad)
    @testset "autodense, out-of-place, $ad" verbose=true showtiming=true begin
        doit_out_of_place(p)
    end
end
