@testitem "doctests" begin
    using OpenMDAOCore
    using Documenter
    doctest(OpenMDAOCore, manual=false)
end

@testitem "VarData" begin
    using OpenMDAOCore
    using Test

    vd = VarData("x", 1.0, (1,), nothing, nothing, nothing, nothing)
    @test vd.name == "x"
    @test vd.val ≈ 1.0
    @test vd.shape == (1,)
    @test vd.units == nothing
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.tags == nothing
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x", 1.0, 1, nothing, nothing, nothing, nothing)
    @test vd.name == "x"
    @test vd.val ≈ 1.0
    @test vd.shape == (1,)
    @test vd.units == nothing
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x")
    @test vd.name == "x"
    @test vd.val ≈ 1.0
    @test vd.shape == (1,)
    @test vd.units == nothing
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; shape=(2, 3))
    @test vd.name == "x"
    @test vd.val ≈ 1.0
    @test vd.shape == (2, 3)
    @test vd.units == nothing
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; val=[3.0, 4.0], shape=(2,))
    @test vd.name == "x"
    @test vd.val ≈ [3.0, 4.0]
    @test vd.shape == (2,)
    @test vd.units == nothing
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; val=[3.0, 4.0], shape=(2,), units="m", tags=["foo", "bar"])
    @test vd.name == "x"
    @test vd.val ≈ [3.0, 4.0]
    @test vd.shape == (2,)
    @test vd.units == "m"
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.tags == ["foo", "bar"]
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; val=[3.0, 4.0], shape=(2,), units="m", tags=["foo", "bar"], shape_by_conn=true, copy_shape="y")
    @test vd.name == "x"
    @test vd.val ≈ [3.0, 4.0]
    @test vd.shape == (2,)
    @test vd.units == "m"
    @test vd.lower == nothing
    @test vd.upper == nothing
    @test vd.tags == ["foo", "bar"]
    @test vd.shape_by_conn == true
    @test vd.copy_shape == "y"

    vd = VarData("x", 1.0, (1,), nothing, -2.0, 2.0,)
    @test vd.name == "x"
    @test vd.val ≈ 1.0
    @test vd.shape == (1,)
    @test vd.units == nothing
    @test vd.lower == -2.0
    @test vd.upper == 2.0
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; val=[3.0, 4.0], shape=(2,), units="m", lower=[-1.0, -2.0], upper=[2.0, 3.0])
    @test vd.name == "x"
    @test vd.val ≈ [3.0, 4.0]
    @test vd.shape == (2,)
    @test vd.units == "m"
    @test vd.lower ≈ [-1.0, -2.0]
    @test vd.upper ≈ [2.0, 3.0]
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    vd = VarData("x"; val=[3.0, 4.0], units="m", lower=[-1.0, -2.0], upper=[2.0, 3.0])
    @test vd.name == "x"
    @test vd.val ≈ [3.0, 4.0]
    @test vd.shape == (2,)
    @test vd.units == "m"
    @test vd.lower ≈ [-1.0, -2.0]
    @test vd.upper ≈ [2.0, 3.0]
    @test vd.shape_by_conn == false
    @test vd.copy_shape == nothing

    @test_throws ArgumentError VarData("x"; val=[3.0, 4.0, 5.0], shape=(2,), units="m", lower=[-1.0, -2.0], upper=[2.0, 3.0])
    @test_throws ArgumentError VarData("x"; val=[3.0, 4.0], shape=(2,), units="m", lower=[-1.0, -2.0, 3.0], upper=[2.0, 3.0])
    @test_throws ArgumentError VarData("x"; val=[3.0, 4.0], shape=(2,), units="m", lower=[-1.0, -2.0], upper=[2.0, 3.0, 10.0])
end

@testitem "PartialsData" begin
    using OpenMDAOCore
    using Test

    pd = PartialsData("y", "x")
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == nothing
    @test pd.cols == nothing
    @test pd.val == nothing
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; method="cs")
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == nothing
    @test pd.cols == nothing
    @test pd.val == nothing
    @test pd.method == "cs"

    pd = PartialsData("y", "x"; rows=[1, 2], cols=[1, 2])
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == [1, 2]
    @test pd.cols == [1, 2]
    @test pd.val == nothing
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; rows=[1, 2], cols=[1, 2], val=[1.0, 2.0])
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == [1, 2]
    @test pd.cols == [1, 2]
    @test pd.val == [1.0, 2.0]
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; val=[1.0  2.0;
                                     3.0  4.0])
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == nothing
    @test pd.cols == nothing
    @test pd.val == [1.0 2.0;
                     3.0 4.0]
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; val=-8.0)
    @test pd.rows == nothing
    @test pd.cols == nothing
    @test pd.val ≈ -8.0
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; rows=[1, 2], cols=[1, 2], val=-8.0)
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test pd.rows == [1, 2]
    @test pd.cols == [1, 2]
    @test pd.val ≈ -8.0
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; rows=0:8-1, cols=fill(0, 8), val=-8.0)
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test all(pd.rows .== [0, 1, 2, 3, 4, 5, 6, 7])
    @test all(pd.cols .== 0)
    @test pd.val ≈ -8.0
    @test pd.method == "exact"

    pd = PartialsData("y", "x"; rows=fill(0, 8), cols=0:8-1, val=-8.0)
    @test pd.of == "y"
    @test pd.wrt == "x"
    @test all(pd.rows .== 0)
    @test all(pd.cols .== [0, 1, 2, 3, 4, 5, 6, 7])
    @test pd.val ≈ -8.0
    @test pd.method == "exact"

    @test_throws ArgumentError PartialsData("y", "x"; rows=[1, 2, 3], cols=[1, 2], val=[1.0, 2.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=[1, 2], cols=[1, 2, 3], val=[1.0, 2.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=[1, 2], cols=[1, 2], val=[1.0, 2.0, 3.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=nothing, cols=[1, 2], val=[1.0, 2.0, 3.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=[1, 2], cols=[1, 2], val=[1.0 2.0;
                                                                                     3.0 4.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=nothing, cols=[1, 2], val=[1.0 2.0;
                                                                                      3.0 4.0])
    @test_throws ArgumentError PartialsData("y", "x"; rows=[1, 2], cols=nothing, val=[1.0 2.0;
                                                                                      3.0 4.0])
end

@testitem "Finding ExplicitComponent methods" begin
    using OpenMDAOCore
    using Test

    struct FooComp1 <: AbstractExplicitComp end
    comp = FooComp1()
    @test !has_compute_partials(comp)
    @test !has_compute_jacvec_product(comp)
    @test !has_setup_partials(comp)

    struct FooComp2 <: AbstractExplicitComp end
    OpenMDAOCore.compute_partials!(self::FooComp2, inputs, partials) = nothing
    comp = FooComp2()
    @test has_compute_partials(comp)
    @test !has_compute_jacvec_product(comp)
    @test !has_setup_partials(comp)

    struct FooComp3 <: AbstractExplicitComp end
    OpenMDAOCore.compute_jacvec_product!(self::FooComp3, inputs, d_inputs, d_outputs, mode) = nothing
    comp = FooComp3()
    @test !has_compute_partials(comp)
    @test has_compute_jacvec_product(comp)
    @test !has_setup_partials(comp)

    struct FooComp4 <: AbstractExplicitComp end
    comp = FooComp4()
    OpenMDAOCore.setup_partials(self::FooComp4, input_sizes, output_sizes) = nothing
    @test !has_compute_partials(comp)
    @test !has_compute_jacvec_product(comp)
    @test has_setup_partials(comp)
end

@testitem "Finding ImplicitComponent methods" begin
    using OpenMDAOCore
    using Test

    struct BarComp1 <: AbstractImplicitComp end
    comp = BarComp1()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp2 <: AbstractImplicitComp end
    OpenMDAOCore.apply_nonlinear!(self::BarComp2, inputs, outputs, residuals) = nothing
    comp = BarComp2()
    @test !has_setup_partials(comp)
    @test has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp3 <: AbstractImplicitComp end
    OpenMDAOCore.solve_nonlinear!(self::BarComp3, inputs, outputs) = nothing
    comp = BarComp3()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp4 <: AbstractImplicitComp end
    OpenMDAOCore.linearize!(self::BarComp4, inputs, outputs, partials) = nothing
    comp = BarComp4()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp5 <: AbstractImplicitComp end
    OpenMDAOCore.apply_linear!(self::BarComp5, inputs, outputs, d_inputs, d_outputs, d_residuals, mode) = nothing
    comp = BarComp5()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp6 <: AbstractImplicitComp end
    OpenMDAOCore.solve_linear!(self::BarComp6, d_outputs, d_residuals, mode) = nothing
    comp = BarComp6()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)

    struct BarComp7 <: AbstractImplicitComp end
    OpenMDAOCore.guess_nonlinear!(self::BarComp7, inputs, outputs, residuals) = nothing
    comp = BarComp7()
    @test !has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test has_guess_nonlinear(comp)

    struct BarComp8 <: AbstractImplicitComp end
    OpenMDAOCore.setup_partials(self::BarComp8, input_sizes, output_sizes) = nothing
    comp = BarComp8()
    @test has_setup_partials(comp)
    @test !has_apply_nonlinear(comp)
    @test !has_solve_nonlinear(comp)
    @test !has_linearize(comp)
    @test !has_apply_linear(comp)
    @test !has_solve_linear(comp)
    @test !has_guess_nonlinear(comp)
end

@testitem "OpenMDAOCore exceptions" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays: ComponentVector
    using Test

    # A dense explicit component created with `force_skip_prep=true`: calling
    # `compute_partials!` before OpenMDAO has set up the problem throws.
    function f_simple!(Y, X, params)
        Y[:y] .= 2.0 .* X[:x]
        return nothing
    end
    comp = create_explicit_component(DenseFlavor(), ADTypes.AutoForwardDiff(), f_simple!,
        ComponentVector(y=0.0), ComponentVector(x=1.0); force_skip_prep=true)
    inputs = Dict("x" => 1.0)
    partials = Dict{Any,Any}()

    @test_throws OpenMDAOPrepNotCreatedError OpenMDAOCore.compute_partials!(comp, inputs, partials)
    @test_throws OpenMDAOCoreError OpenMDAOCore.compute_partials!(comp, inputs, partials)

    # The error message names the method and explains the cause.
    err = try
        OpenMDAOCore.compute_partials!(comp, inputs, partials)
    catch e
        e
    end
    @test err isa OpenMDAOPrepNotCreatedError
    msg = sprint(Base.showerror, err)
    @test occursin("compute_partials!", msg)
    @test occursin("force_skip_prep", msg)
    @test occursin("ADExplicitComp", msg)

    # Same for the `linearize!` of a dense implicit component.
    function f_implicit!(R, Y, X, params)
        R[:r] .= X[:x] .- Y[:y]
        return nothing
    end
    icomp = create_implicit_component(DenseFlavor(), true, ADTypes.AutoForwardDiff(), f_implicit!,
        ComponentVector(y=0.0), ComponentVector(x=1.0); force_skip_prep=true)
    outputs = Dict("y" => 0.0)

    @test_throws OpenMDAOPrepNotCreatedError OpenMDAOCore.linearize!(icomp, inputs, outputs, partials)

    # Backwards compatibility: `OpenMDAOMethodError` is still thrown by the
    # interface fallbacks and is catchable directly, and both exception types
    # are subtypes of the `OpenMDAOCoreError` root.
    struct NoMethodsComp <: OpenMDAOCore.AbstractExplicitComp end
    @test_throws OpenMDAOMethodError OpenMDAOCore.compute!(NoMethodsComp(), inputs, Dict{String,Any}())
    @test OpenMDAOMethodError <: OpenMDAOCoreError
    @test OpenMDAOPrepNotCreatedError <: OpenMDAOCoreError
end

@testitem "get_rows_cols" begin
    using OpenMDAOCore
    using Test

    ss_sizes = Dict(:i=>2, :j=>3, :k=>4)

    of_ss = [:i]
    wrt_ss = [:i]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1])
    @test all(cols .== [0, 1])

    of_ss = [:i]
    wrt_ss = [:j]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 1, 1, 1])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:i]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 0, 0, 1, 1, 1])

    of_ss = [:i, :j]
    wrt_ss = [:j]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:j, :i]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 2, 4, 1, 3, 5])

    of_ss = [:i, :j]
    wrt_ss = [:j, :k]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    ss_sizes = Dict(:k=>2, :i=>3, :j=>4)

    of_ss = [:k, :i]
    wrt_ss = [:i, :j]
    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
end

@testitem "_get_rows_cols_dict_from_sparsity, 0d, 1d, 2d" begin
    using Test
    using ComponentArrays: ComponentVector, ComponentMatrix, getaxes
    using SparseArrays: sparse
    using SparseMatrixColorings: SparseMatrixColorings
    using OpenMDAOCore: OpenMDAOCore

    N, M = 3, 4
    X_ca = ComponentVector(a=0.0, b=zeros(Float64, N), c=zeros(Float64, N, M), crev=zeros(Float64, M, N))
    Y_ca = ComponentVector(d=zeros(Float64, N), e=zeros(Float64, M), f=zeros(Float64, N, M), g=0.0, frev=zeros(Float64, M, N), h=0.0)
    J_ca = Y_ca .* X_ca'

    # Define the sparsity.
    J_ca .= 0.0
    for i = 1:N
        @view(J_ca[:d, :a])[i] = 1.0
        @view(J_ca[:d, :b])[i, i] = 1.0
        for j in 1:M
            @view(J_ca[:d, :c])[i, i, j] = 1.0
            @view(J_ca[:d, :crev])[i, j, i] = 1.0
            @view(J_ca[:f, :a])[i, j] = 1.0
            @view(J_ca[:f, :b])[i, j, i] = 1.0
            @view(J_ca[:f, :c])[i, j, i, j] = 1.0
            @view(J_ca[:f, :crev])[i, j, j, i] = 1.0
            @view(J_ca[:frev, :a])[j, i] = 1.0
            @view(J_ca[:frev, :b])[j, i, i] = 1.0
            @view(J_ca[:frev, :c])[j, i, i, j] = 1.0
            @view(J_ca[:frev, :crev])[j, i, j, i] = 1.0
            @view(J_ca[:g, :c])[i, j] = 1.0
            @view(J_ca[:g, :crev])[j, i] = 1.0
        end
        @view(J_ca[:g, :b])[i] = 1.0
    end

    for j in 1:M
        @view(J_ca[:e, :a])[j] = 1.0
        for i = 1:N
            @view(J_ca[:e, :b])[j, i] = 1.0
            @view(J_ca[:e, :c])[j, i, j] = 1.0
            @view(J_ca[:e, :crev])[j, j, i] = 1.0
        end
    end

    @view(J_ca[:g, :a])[1] = 1.0

    # Create a sparse version of J_ca.
    J_ca_sparse = ComponentMatrix(sparse(J_ca), getaxes(J_ca))

    # Get the rows and cols dict.
    rcdict = OpenMDAOCore._get_rows_cols_dict_from_sparsity(J_ca_sparse)

    # Shouldn't make a difference if the Jacobian is sparse or not.
    rcdict_not_sparse_J_ca = OpenMDAOCore._get_rows_cols_dict_from_sparsity(J_ca)
    @test rcdict == rcdict_not_sparse_J_ca

    # Short function that puts the rows and cols in a standard order for comparison purposes.
    function rows_cols_normalize(rows, cols)
        out = sortslices(hcat(rows, cols); dims=1)
        return out[:, 1], out[:, 2]
    end

    ss_sizes = Dict(:i=>N, :j=>M, :s=>1)

    rows, cols = rows_cols_normalize(rcdict[:d, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:d, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:d, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:d, :crev]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :crev]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :crev]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:frev, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:frev, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:frev, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:frev, :crev]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :crev]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rcdict[:h, :a]
    @test rows == Vector{Int}()
    @test cols == Vector{Int}()
end

@testitem "_get_rows_cols_dict_from_sparsity, 3d" begin
    using Test
    using ComponentArrays: ComponentVector, ComponentMatrix, getaxes
    using SparseArrays: sparse
    using SparseMatrixColorings: SparseMatrixColorings
    using OpenMDAOCore
    using OpenMDAOCore: _get_rows_cols_dict_from_sparsity

    I, J, K = 3, 4, 5
    X_ca = ComponentVector(a=zeros(Float64, I), b=zeros(Float64, I, J), c=zeros(Float64, J, I, K))
    Y_ca = ComponentVector(d=zeros(Float64, I), e=zeros(Float64, J), f=zeros(Float64, I, J), g=zeros(Float64, K, J, I))
    J_ca = Y_ca .* X_ca'

    # Define the sparsity.
    J_ca .= 0.0
    for i = 1:I
        @view(J_ca[:d, :a])[i, i] = 1.0
        for j in 1:J
            @view(J_ca[:d, :b])[i, i, j] = 1.0

            @view(J_ca[:e, :a])[j, i] = 1.0
            @view(J_ca[:e, :b])[j, i, j] = 1.0

            @view(J_ca[:f, :a])[i, j, i] = 1.0
            @view(J_ca[:f, :b])[i, j, i, j] = 1.0

            for k in 1:K
                @view(J_ca[:d, :c])[i, j, i, k] = 1.0
                @view(J_ca[:e, :c])[j, j, i, k] = 1.0
                @view(J_ca[:f, :c])[i, j, j, i, k] = 1.0

                @view(J_ca[:g, :a])[k, j, i, i] = 1.0
                @view(J_ca[:g, :b])[k, j, i, i, j] = 1.0
                @view(J_ca[:g, :c])[k, j, i, j, i, k] = 1.0
            end
        end
    end

    # Create a sparse version of J_ca.
    J_ca_sparse = ComponentMatrix(sparse(J_ca), getaxes(J_ca))

    # Get the rows and cols dict.
    rcdict = _get_rows_cols_dict_from_sparsity(J_ca_sparse)

    rcdict_not_sparse_J_ca = _get_rows_cols_dict_from_sparsity(J_ca)
    @test rcdict == rcdict_not_sparse_J_ca

    # Short function that puts the rows and cols in a standard order for comparison purposes.
    function rows_cols_normalize(rows, cols)
        out = sortslices(hcat(rows, cols); dims=1)
        return out[:, 1], out[:, 2]
    end

    ss_sizes = Dict(:i=>I, :j=>J, :k=>K)

    rows, cols = rows_cols_normalize(rcdict[:d, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:d, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:d, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:e, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:f, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :a]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :b]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)

    rows, cols = rows_cols_normalize(rcdict[:g, :c]...)
    rows_check, cols_check = rows_cols_normalize(get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
    @test all(rows .== rows_check)
    @test all(cols .== cols_check)
end

@testitem "_resize_component_vector preserves key order" begin
    using OpenMDAOCore
    using Test
    using ComponentArrays: ComponentVector

    # Use non-alphabetical key order to catch any sorting/reordering bugs.
    X_ca = ComponentVector(z=1.0, a=[8.0], m=reshape(collect(1:3), 3, 1))
    original_keys = collect(keys(X_ca))

    # Resize a subset of the components; leave the rest untouched.
    sizes = Dict(:a => 5, :m => (3, 4))
    X_ca_resized = OpenMDAOCore._resize_component_vector(X_ca, sizes)

    # 1. Key order must match the original, exactly.
    @test collect(keys(X_ca_resized)) == original_keys

    # 2. Resized components got the new shapes; untouched ones kept theirs.
    @test size(X_ca_resized.a) == (5,)
    @test size(X_ca_resized.m) == (3, 4)
    @test size(X_ca_resized.z) == ()

    # 3. Existing values were copied over (where they still fit).
    @test all(X_ca_resized.a .≈ X_ca.a)
    @test all(X_ca_resized.m .≈ X_ca.m)
    @test only(X_ca_resized.z) ≈ 1.0
end

@testitem "create_explicit_component" begin
    using OpenMDAOCore
    using Test
    using ADTypes: ADTypes
    using ForwardDiff: ForwardDiff
    using ComponentArrays: ComponentVector

    # Simple in-place function.
    f!(Y, X, params) = (Y.y = X.x ^ 2; nothing)
    f(X, params) = ComponentVector(y=X.x ^ 2)
    X_ca = ComponentVector(x=2.0)
    Y_ca = ComponentVector(y=0.0)

    # Dense flavor: any backend works; returns the right flavor/InPlace.
    comp = create_explicit_component(DenseFlavor(), ADTypes.AutoForwardDiff(), f!, Y_ca, X_ca)
    @test comp isa ADExplicitComp{DenseFlavor, true}

    comp_oop = create_explicit_component(DenseFlavor(), ADTypes.AutoForwardDiff(), f, X_ca)
    @test comp_oop isa ADExplicitComp{DenseFlavor, false}

    # Matrix-free forward flavor with a forward-capable backend is OK.
    comp_mf = create_explicit_component(MatrixFreeForwardFlavor(), ADTypes.AutoForwardDiff(), f!, Y_ca, X_ca)
    @test comp_mf isa ADExplicitComp{MatrixFreeForwardFlavor, true}

    # Matrix-free forward flavor with a reverse-only backend must error.
    # AutoZygote is reverse-mode (VJP only).
    using Zygote: Zygote
    @test_throws ArgumentError create_explicit_component(
        MatrixFreeForwardFlavor(), ADTypes.AutoZygote(), f!, Y_ca, X_ca)
end
