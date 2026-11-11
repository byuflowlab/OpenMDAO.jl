@testitem "implicit sparse automatic, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitAutomaticTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit sparse automatic, in-place, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitAutomaticTestPrep(4, 3, "reversediff"))
end

@testitem "implicit sparse automatic, in-place, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitAutomaticTestPrep(4, 3, "enzymeforward"))
end

@testitem "implicit sparse automatic, in-place, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitAutomaticTestPrep(4, 3, "enzymereverse"))
end

@testitem "implicit sparse automatic, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitAutomaticTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit sparse automatic, out-of-place, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitAutomaticTestPrep(4, 3, "reversediff"))
end

@testitem "implicit sparse automatic, out-of-place, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitAutomaticTestPrep(4, 3, "zygote"))
end

@testitem "implicit sparse automatic, in-place, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitAutomaticShapeByConnTestPrep(4, 3, "forwarddiff"))
end
