@testitem "implicit matrix-free, in-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

@testitem "implicit matrix-free, out-of-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

@testitem "implicit matrix-free, in-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_in_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end

@testitem "implicit matrix-free, out-of-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end

@testitem "implicit matrix-free, in-place, forward, forwarddiff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit matrix-free, in-place, reverse, reversediff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "implicit matrix-free, in-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "implicit matrix-free, out-of-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_out_of_place_fwd_rev_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "implicit matrix-free, in-place, fwd+rev, forwarddiff+reversediff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev_sbc_implicit(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "forwarddiff"), "reversediff")
end

@testitem "implicit matrix-free, guess_nonlinear, in-place, forwarddiff" setup=[ADCallbacks] begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using Test

    # `M` and `N` will be passed via the params argument.
    M = 4
    N = 3
    params = (M, N)
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))
    X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # In-place guess function: sets the states to their exact values given the
    # inputs, and zeroes the residuals (though OpenMDAO treats the residuals
    # as read-only in `guess_nonlinear`, so the zeroing has no effect).
    # Uses the same residual function as the regular matrix-free tests to
    # compute what the residuals should be.
    function f_guess!(Y, R, X, params)
        f_implicit!(R, Y, X, params)
        Y[:e] .= @view R[:e]
        Y[:f] .= @view R[:f]
        Y[:g] .= @view R[:g]
        f_implicit!(R, Y, X, params)
        return nothing
    end

    comp = create_implicit_component(MatrixFreeForwardFlavor(), true, ADTypes.AutoForwardDiff(), f_implicit!, Y_ca, X_ca;
        params=params, guess_nonlinear=f_guess!)

    @test comp isa ADImplicitComp
    @test comp.guess_func === f_guess!
    @test has_guess_nonlinear(comp) == true

    inputs = ca2strdict(X_ca)
    outputs = ca2strdict(Y_ca)
    outputs["e"] .= 0.0
    residuals = ca2strdict(ComponentVector(e=ones(N), f=ones(M, N), g=ones(N, M)))

    OpenMDAOCore.guess_nonlinear!(comp, inputs, outputs, residuals)

    # The states should now hold their exact values. The residuals are
    # read-only in `guess_nonlinear`, so they are unchanged.
    @test all(residuals["e"] .≈ 1.0)
    @test all(residuals["f"] .≈ 1.0)
    @test all(residuals["g"] .≈ 1.0)
end
