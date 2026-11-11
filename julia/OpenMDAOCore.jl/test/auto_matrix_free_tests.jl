@testitem "auto matrix-free, in-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, in-place, forward, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeTestPrep(4, 3, "enzymeforward", false))
end

@testitem "auto matrix-free, in-place, forward, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, in-place, forward, shape_by_conn, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "enzymeforward", false))
end

@testitem "auto matrix-free, in-place, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, in-place, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, in-place, reverse, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "enzymereverse", false))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "enzymereverse", false))
end

@testitem "auto matrix-free, out-of-place, forward, forwarddiff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_forward(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, out-of-place, forward, shape_by_conn, forwarddiff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, out-of-place, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, out-of-place, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, out-of-place, reverse, zygote, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "zygote", false))
end

@testitem "auto matrix-free, out-of-place, reverse, zygote, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "zygote", true))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, zygote, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "zygote", false))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, zygote, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "zygote", true))
end

@testitem "auto matrix-free, in-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "auto matrix-free, out-of-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_out_of_place_fwd_rev(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "auto matrix-free, in-place, fwd+rev, forwarddiff+reversediff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev_sbc(AutoMatrixFreeShapeByConnTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "auto matrix-free, fwd+rev, backend validation errors" setup=[ADCallbacks] begin
    using ADTypes: ADTypes
    prep = AutoMatrixFreeTestPrep(4, 3, "reversediff", false)
    # AutoReverseDiff can't do JVPs, so it is not a valid forward backend.
    @test_throws ArgumentError create_explicit_component(MatrixFreeForwardReverseFlavor(),
        prep.ad_backend, ADTypes.AutoReverseDiff(), f_simple!, prep.Y_ca, prep.X_ca; params=(4, 3))
    # AutoForwardDiff can't do VJPs, so it is not a valid reverse backend.
    @test_throws ArgumentError create_implicit_component(MatrixFreeForwardReverseFlavor(), true,
        ADTypes.AutoForwardDiff(), ADTypes.AutoForwardDiff(), f_implicit!, prep.Y_ca, prep.X_ca)
end
