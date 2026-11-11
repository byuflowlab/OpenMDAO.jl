@testitem "autosparse_manual, in-place, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseManualTestPrep(4, 3, "forwarddiff"))
end

@testitem "autosparse_manual, in-place, reversediff" setup=[ADCallbacks] begin
    doit_in_place(AutosparseManualTestPrep(4, 3, "reversediff"))
end

@testitem "autosparse_manual, in-place, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place(AutosparseManualTestPrep(4, 3, "enzymeforward"))
end

@testitem "autosparse_manual, in-place, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place(AutosparseManualTestPrep(4, 3, "enzymereverse"))
end

@testitem "autosparse_manual, out-of-place, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseManualTestPrep(4, 3, "forwarddiff"))
end

@testitem "autosparse_manual, out-of-place, reversediff" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseManualTestPrep(4, 3, "reversediff"))
end

@testitem "autosparse_manual, out-of-place, zygote" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseManualTestPrep(4, 3, "zygote"))
end
