@testitem "autosparse_automatic, in-place, forwarddiff, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "forwarddiff", :direct))
end

@testitem "autosparse_automatic, in-place, reversediff, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "reversediff", :direct))
end

@testitem "autosparse_automatic, in-place, enzymeforward, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "enzymeforward", :direct))
end

@testitem "autosparse_automatic, in-place, enzymereverse, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "enzymereverse", :direct))
end

@testitem "autosparse_automatic, in-place, forwarddiff, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "forwarddiff", :iterative))
end

@testitem "autosparse_automatic, in-place, reversediff, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "reversediff", :iterative))
end

@testitem "autosparse_automatic, in-place, enzymeforward, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "enzymeforward", :iterative))
end

@testitem "autosparse_automatic, in-place, enzymereverse, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticTestPrep(4, 3, "enzymereverse", :iterative))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, forwarddiff, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "forwarddiff", :direct))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, reversediff, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "reversediff", :direct))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, enzymeforward, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "enzymeforward", :direct))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, enzymereverse, direct" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "enzymereverse", :direct))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, forwarddiff, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "forwarddiff", :iterative))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, reversediff, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "reversediff", :iterative))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, enzymeforward, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "enzymeforward", :iterative))
end

@testitem "autosparse_automatic, in-place, shape_by_conn, enzymereverse, iterative" setup=[ADCallbacks] begin
    doit_in_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "enzymereverse", :iterative))
end

@testitem "autosparse_automatic, out-of-place, forwarddiff, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "forwarddiff", :direct))
end

@testitem "autosparse_automatic, out-of-place, reversediff, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "reversediff", :direct))
end

@testitem "autosparse_automatic, out-of-place, zygote, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "zygote", :direct))
end

@testitem "autosparse_automatic, out-of-place, forwarddiff, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "forwarddiff", :iterative))
end

@testitem "autosparse_automatic, out-of-place, reversediff, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "reversediff", :iterative))
end

@testitem "autosparse_automatic, out-of-place, zygote, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticTestPrep(4, 3, "zygote", :iterative))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, forwarddiff, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "forwarddiff", :direct))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, reversediff, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "reversediff", :direct))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, zygote, direct" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "zygote", :direct))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, forwarddiff, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "forwarddiff", :iterative))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, reversediff, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "reversediff", :iterative))
end

@testitem "autosparse_automatic, out-of-place, shape_by_conn, zygote, iterative" setup=[ADCallbacks] begin
    doit_out_of_place(AutosparseAutomaticShapeByConnTestPrep(4, 3, "zygote", :iterative))
end
