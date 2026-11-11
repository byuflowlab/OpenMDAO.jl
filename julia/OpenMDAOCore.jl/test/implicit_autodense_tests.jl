@testitem "implicit dense, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_implicit(AutoDenseImplicitTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit dense, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place_implicit(AutoDenseImplicitTestPrep(4, 3, "forwarddiff"))
end