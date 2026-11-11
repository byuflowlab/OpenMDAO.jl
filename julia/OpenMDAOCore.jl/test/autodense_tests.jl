@testitem "autodense, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, in-place, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, in-place, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "enzymeforward"))
end

@testitem "autodense, in-place, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseTestPrep(4, 3, "enzymereverse"))
end

@testitem "autodense, in-place, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, in-place, shape_by_conn, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, in-place, shape_by_conn, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "enzymeforward"))
end

@testitem "autodense, in-place, shape_by_conn, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutoDenseShapeByConnTestPrep(4, 3, "enzymereverse"))
end

@testitem "autodense, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, out-of-place, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, out-of-place, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseTestPrep(4, 3, "zygote"))
end

@testitem "autodense, out-of-place, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "autodense, out-of-place, shape_by_conn, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "autodense, out-of-place, shape_by_conn, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutoDenseShapeByConnTestPrep(4, 3, "zygote"))
end

@testitem "ADExplicitComp create, flavor/backend validation" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ForwardDiff
    using SparseMatrixColorings: SparseMatrixColorings
    using Test

    function f_simple!(Y, X, params)
        Y[:y] = only(X[:x])^2
        return nothing
    end

    function f_simple(X, params)
        return ComponentVector(y=only(X[:x])^2)
    end

    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(
        ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    auto_sparse_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff();
                                             sparsity_detector=sparsity_detector,
                                             coloring_algorithm=coloring_algorithm)

    # SparseFlavor requires an AutoSparse backend.
    @test_throws ArgumentError create_explicit_component(SparseFlavor(), ADTypes.AutoForwardDiff(), f_simple!, Y_ca, X_ca)
    @test_throws ArgumentError create_explicit_component(SparseFlavor(), ADTypes.AutoForwardDiff(), f_simple, X_ca)
    # DenseFlavor requires a dense (non-AutoSparse) backend.
    @test_throws ArgumentError create_explicit_component(DenseFlavor(), auto_sparse_backend, f_simple!, Y_ca, X_ca)
    @test_throws ArgumentError create_explicit_component(DenseFlavor(), auto_sparse_backend, f_simple, X_ca)
    # The compatible combinations work.
    comp_dense = create_explicit_component(DenseFlavor(), ADTypes.AutoForwardDiff(), f_simple!, Y_ca, X_ca)
    @test comp_dense isa ADExplicitComp{DenseFlavor}
    comp_sparse = create_explicit_component(SparseFlavor(), auto_sparse_backend, f_simple, X_ca)
    @test comp_sparse isa ADExplicitComp{SparseFlavor}
end
