# Tests and benchmarks for the prepared copying machinery (ported from
# OpenMDAO4Core.jl's `sparse_utils_tests.jl`). The `@allocated` assertions
# verify that the prepared copies are allocation-free in steady state, and the
# `@benchmark` blocks print timings comparing them against the key-based
# ("old") implementations.

@testitem "prepared input copying is allocation-free" setup=[CopyUtils] begin
    data = zeros(Float64, 5)
    names = ["a", "b"]
    ranges = [1:1, 2:5]
    inputs = Dict{String, Vector{Float64}}(
        "a" => [3.0],
        "b" => [4.0, 5.0, 6.0, 7.0],
    )

    # Warm up the exact call before measuring steady-state allocations.
    _copy_component_inputs!(data, inputs, names, ranges)
    data .= 0.0

    allocations = @allocated _copy_component_inputs!(data, inputs, names, ranges)

    @test allocations == 0
    @test data == [3.0, 4.0, 5.0, 6.0, 7.0]
end

@testitem "prepared input copying handles multidimensional inputs" setup=[CopyUtils] begin
    input_a = reshape(collect(1.0:6.0), 2, 3)
    input_b = reshape(collect(11.0:22.0), 2, 2, 3)
    inputs = Dict{String, Union{Array{Float64, 2}, Array{Float64, 3}}}(
        "a" => input_a,
        "b" => input_b,
    )
    names = ["a", "b"]
    ranges = [1:length(input_a), (length(input_a) + 1):(length(input_a) + length(input_b))]
    target = ComponentVector(a=zeros(2, 3), b=zeros(2, 2, 3))
    expected = vcat(vec(input_a), vec(input_b))
    expected_c_order = vcat(vec(permutedims(input_a, (2, 1))), vec(permutedims(input_b, (3, 2, 1))))

    # Warm up the exact call before measuring steady-state allocations.
    _copy_component_inputs!(getdata(target), inputs, names, ranges)
    getdata(target) .= 0.0

    allocations = @allocated _copy_component_inputs!(getdata(target), inputs, names, ranges)

    @test allocations == 0
    @test getdata(target) == expected

    positional_inputs = (input_a, input_b)
    getdata(target) .= 0.0
    _copy_component_inputs!(getdata(target), positional_inputs, ranges)
    getdata(target) .= 0.0
    positional_allocations = @allocated _copy_component_inputs!(
        getdata(target), positional_inputs, ranges,
    )

    @test positional_allocations == 0
    @test getdata(target) == expected
    @test getdata(target) != expected_c_order
    @test target.a == input_a
    @test target.b == input_b
end

@testitem "component vector output copying is allocation-free" setup=[CopyUtils] begin
    data = collect(1.0:4.0)
    output_prep = OpenMDAOCore.InputCopyPrep(ComponentVector(x=zeros(4)))
    outputs = Dict("x" => zeros(4))

    _copy_component_outputs!(data, outputs, output_prep)
    @test outputs["x"] == data

    outputs["x"] .= 1.0
    _copy_component_outputs_accum!(data, outputs, output_prep)
    @test outputs["x"] == data .+ 1.0

    outputs["x"] .= 0.0
    _copy_component_outputs!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs!(data, outputs, output_prep)
    @test allocations == 0

    outputs["x"] .= 0.0
    _copy_component_outputs_accum!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs_accum!(data, outputs, output_prep)
    @test allocations == 0
end

@testitem "multidimensional component output copying is allocation-free and layout-independent" setup=[CopyUtils] begin
    data = collect(1.0:6.0)
    output_prep = OpenMDAOCore.InputCopyPrep(ComponentVector(x=zeros(2, 3)))
    outputs = Dict("x" => permutedims(zeros(3, 2), (2, 1)))
    expected = reshape(data, 2, 3)

    _copy_component_outputs!(data, outputs, output_prep)
    @test outputs["x"] == expected

    outputs["x"] .= 1.0
    _copy_component_outputs_accum!(data, outputs, output_prep)
    @test outputs["x"] == expected .+ 1.0

    outputs["x"] .= 0.0
    _copy_component_outputs!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs!(data, outputs, output_prep)
    @test allocations == 0

    outputs["x"] .= 0.0
    _copy_component_outputs_accum!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs_accum!(data, outputs, output_prep)
    @test allocations == 0
end

@testitem "scalar component output copying is allocation-free" setup=[CopyUtils] begin
    data = [23.0]
    output_prep = OpenMDAOCore.InputCopyPrep(ComponentVector(x=0.0))
    outputs = Dict("x" => 0.0)

    _copy_component_outputs!(data, outputs, output_prep)
    @test outputs["x"] == 23.0

    _copy_component_outputs_accum!(data, outputs, output_prep)
    @test outputs["x"] == 46.0

    _copy_component_outputs!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs!(data, outputs, output_prep)
    @test allocations == 0

    _copy_component_outputs_accum!(data, outputs, output_prep)
    allocations = @allocated _copy_component_outputs_accum!(data, outputs, output_prep)
    @test allocations == 0
end

@testitem "prepared copying uses fewer allocations than key-based copying" setup=[CopyUtils] begin
    source = ComponentVector(a=collect(1.0:32.0), b=collect(33.0:96.0))
    inputs = Dict("a" => copy(source.a), "b" => copy(source.b))
    input_data = zeros(96)
    input_prep = OpenMDAOCore.InputCopyPrep(source)

    old_copy_inputs!(source, inputs)
    _copy_component_inputs!(input_data, inputs, input_prep)
    old_input_allocations = @allocated old_copy_inputs!(source, inputs)
    new_input_allocations = @allocated _copy_component_inputs!(input_data, inputs, input_prep)
    @test new_input_allocations == 0
    @test new_input_allocations < old_input_allocations

    outputs_old = Dict("a" => zeros(32), "b" => zeros(64))
    outputs_new = Dict("a" => zeros(32), "b" => zeros(64))
    output_data = getdata(source)
    output_prep = OpenMDAOCore.InputCopyPrep(source)
    old_copy_outputs!(source, outputs_old)
    _copy_component_outputs!(output_data, outputs_new, output_prep)
    old_output_allocations = @allocated old_copy_outputs!(source, outputs_old)
    new_output_allocations = @allocated _copy_component_outputs!(output_data, outputs_new, output_prep)
    @test new_output_allocations == 0
    @test new_output_allocations < old_output_allocations

    old_copy_outputs_accum!(source, outputs_old)
    _copy_component_outputs_accum!(output_data, outputs_new, output_prep)
    old_accum_allocations = @allocated old_copy_outputs_accum!(source, outputs_old)
    new_accum_allocations = @allocated _copy_component_outputs_accum!(output_data, outputs_new, output_prep)
    @test new_accum_allocations == 0
    @test new_accum_allocations < old_accum_allocations

    println("key-based vs prepared copying timings:")
    println("old input copying:")
    display(@benchmark old_copy_inputs!($source, $inputs) samples=10)
    println("new input copying:")
    display(@benchmark _copy_component_inputs!($input_data, $inputs, $input_prep) samples=10)
    println("old output copying:")
    display(@benchmark old_copy_outputs!($source, $outputs_old) samples=10)
    println("new output copying:")
    display(@benchmark _copy_component_outputs!($output_data, $outputs_new, $output_prep) samples=10)
    println("old output copying with accumulation:")
    display(@benchmark old_copy_outputs_accum!($source, $outputs_old) samples=10)
    println("new output copying with accumulation:")
    display(@benchmark _copy_component_outputs_accum!($output_data, $outputs_new, $output_prep) samples=10)

    dense_J_ca = ComponentMatrix(
        reshape(collect(1.0:4.0), 4, 1),
        (only(getaxes(ComponentVector(y=zeros(2, 2)))), only(getaxes(ComponentVector(a=zeros(1))))),
    )
    dense_partials_old = Dict{Tuple{String,String}, Matrix{Float64}}(("y", "a") => zeros(2, 2))
    dense_partials_new = Dict{Tuple{String,String}, Matrix{Float64}}(("y", "a") => zeros(2, 2))
    dense_prep = OpenMDAOCore.DensePartialCopyPrep(dense_J_ca)

    old_dense_partials!(dense_partials_old, dense_J_ca)
    _copy_dense_partials!(dense_partials_new, dense_J_ca, dense_prep)
    old_dense_allocations = @allocated old_dense_partials!(dense_partials_old, dense_J_ca)
    new_dense_allocations = @allocated _copy_dense_partials!(dense_partials_new, dense_J_ca, dense_prep)
    @test new_dense_allocations == 0
    @test new_dense_allocations < old_dense_allocations
    @test dense_partials_new == dense_partials_old

    println("key-based vs prepared dense partial copying timings:")
    println("old dense partials copying:")
    display(@benchmark old_dense_partials!($dense_partials_old, $dense_J_ca) samples=10)
    println("new dense partials copying:")
    display(@benchmark _copy_dense_partials!($dense_partials_new, $dense_J_ca, $dense_prep) samples=10)

    J_ca = ComponentMatrix(
        sparse([1, 1, 2], [1, 3, 4], [10.0, 30.0, 40.0], 2, 4),
        (only(getaxes(ComponentVector(y=zeros(2)))), only(getaxes(ComponentVector(a=0.0, b=zeros(3))))),
    )
    partials_old = Dict{Tuple{String,String}, Vector{Float64}}(
        ("y", "a") => zeros(1),
        ("y", "b") => zeros(2),
    )
    partials_new = Dict{Tuple{String,String}, Vector{Float64}}(
        ("y", "a") => zeros(1),
        ("y", "b") => zeros(2),
    )
    raxis, caxis = getaxes(J_ca)
    rcdict = Dict((:y, :a) => ([1], [1]), (:y, :b) => ([1, 2], [2, 3]))
    blocks = [
        SparsePartialBlock("y", "a", [1], [1]),
        SparsePartialBlock("y", "b", [1, 2], [3, 4]),
    ]
    old_sparse_assembly!(J_ca, partials_old, raxis, caxis, rcdict)
    _assemble_sparse_partials!(partials_new, J_ca, blocks)
    old_sparse_allocations = @allocated old_sparse_assembly!(J_ca, partials_old, raxis, caxis, rcdict)
    new_sparse_allocations = @allocated _assemble_sparse_partials!(partials_new, J_ca, blocks)
    @test new_sparse_allocations == 0
    @test new_sparse_allocations < old_sparse_allocations
    @test partials_new == partials_old

    println("old sparse Jacobian assembly:")
    display(@benchmark old_sparse_assembly!($J_ca, $partials_old, $raxis, $caxis, $rcdict) samples=10)
    println("new sparse Jacobian assembly:")
    display(@benchmark _assemble_sparse_partials!($partials_new, $J_ca, $blocks) samples=10)
end

@testitem "sparse partial assembly is allocation-free" setup=[CopyUtils] begin
    inputs_ca = ComponentVector(a=0.0, b=zeros(3))
    outputs_ca = ComponentVector(y=zeros(2))
    J_ca = ComponentMatrix(
        sparse([1, 1, 2], [1, 2, 4], [10.0, 20.0, 30.0], 2, 4),
        (only(getaxes(outputs_ca)), only(getaxes(inputs_ca))),
    )
    partials = Dict{Tuple{String, String}, Vector{Float64}}(
        ("y", "a") => zeros(1),
        ("y", "b") => zeros(2),
    )
    blocks = [
        SparsePartialBlock("y", "a", [1], [1]),
        SparsePartialBlock("y", "b", [1, 2], [2, 4]),
    ]

    # Warm up the exact call before measuring steady-state allocations.
    _assemble_sparse_partials!(partials, J_ca, blocks)
    for values in values(partials)
        values .= 0.0
    end

    allocations = @allocated _assemble_sparse_partials!(partials, J_ca, blocks)

    @test allocations == 0
    @test partials["y", "a"] == [10.0]
    @test partials["y", "b"] == [20.0, 30.0]
end
