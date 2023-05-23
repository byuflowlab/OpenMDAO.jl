using OpenMDAOCore
using Test
using Documenter

doctest(OpenMDAOCore, manual=false)

@testset "VarData" begin
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

@testset "PartialsData" begin
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

@testset "Finding ExplicitComponent methods" begin
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

@testset "Finding ImplicitComponent methods" begin
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

@testset "get_rows_cols" begin
    ss_sizes = Dict(:i=>2, :j=>3, :k=>4)

    of_ss = [:i]
    wrt_ss = [:i]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1])
    @test all(cols .== [0, 1])

    of_ss = [:i]
    wrt_ss = [:j]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 1, 1, 1])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:i]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 0, 0, 1, 1, 1])

    of_ss = [:i, :j]
    wrt_ss = [:j]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:j, :i]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 2, 4, 1, 3, 5])

    of_ss = [:i, :j]
    wrt_ss = [:j, :k]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    ss_sizes = Dict(:k=>2, :i=>3, :j=>4)

    of_ss = [:k, :i]
    wrt_ss = [:i, :j]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
end
