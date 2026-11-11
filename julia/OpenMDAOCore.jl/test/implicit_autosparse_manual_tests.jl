@testitem "implicit sparse manual, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitManualTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit sparse manual, in-place, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitManualTestPrep(4, 3, "reversediff"))
end

@testitem "implicit sparse manual, in-place, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitManualTestPrep(4, 3, "enzymeforward"))
end

@testitem "implicit sparse manual, in-place, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutosparseImplicitManualTestPrep(4, 3, "enzymereverse"))
end

@testitem "implicit sparse manual, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitManualTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit sparse manual, out-of-place, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitManualTestPrep(4, 3, "reversediff"))
end

@testitem "implicit sparse manual, out-of-place, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseImplicitManualTestPrep(4, 3, "zygote"))
end
