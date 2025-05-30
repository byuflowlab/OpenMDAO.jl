using Test: @testset
using SafeTestsets: @safetestset

@safetestset "doctests" begin
    using OpenMDAOCore
    using Documenter
    doctest(OpenMDAOCore, manual=false)
end

@safetestset "VarData" begin
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

@safetestset "PartialsData" begin
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

@safetestset "Finding ExplicitComponent methods" begin
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

@safetestset "Finding ImplicitComponent methods" begin
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

@safetestset "get_rows_cols" begin
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

@safetestset "get_rows_cols_dict_from_sparsity" begin

    @safetestset "0d, 1d, 2d" begin
        using OpenMDAOCore
        using Test
        using ComponentArrays: ComponentVector, ComponentMatrix, getaxes
        using SparseArrays: sparse

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
        rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

        # Shouldn't make a difference if the Jacobian is sparse or not.
        rcdict_not_sparse_J_ca = get_rows_cols_dict_from_sparsity(J_ca)
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

    @safetestset "3d" begin
        using OpenMDAOCore
        using Test
        using ComponentArrays: ComponentVector, ComponentMatrix, getaxes
        using SparseArrays: sparse

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
        rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

        rcdict_not_sparse_J_ca = get_rows_cols_dict_from_sparsity(J_ca)
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

end

@safetestset "unitfulify" begin
    include("unitfulify.jl")
end

@safetestset "Aviary utils" begin
    include("aviary_utils.jl")
end

@testset "SparseADExplicitComp" begin
    @safetestset "manual sparsity" begin
        include("autosparse_manual.jl")
    end
    @safetestset "automatic sparsity" begin
        include("autosparse_automatic.jl")
    end
    @safetestset "automatic sparsity, with Aviary metadata" begin
        include("autosparse_automatic_aviary.jl")
    end
end

@testset "MatrixFreeADExplicitComp" begin
    @safetestset "in-place" begin
        include("auto_matrix_free_in_place.jl")
    end
    @safetestset "in-place, with Aviary metadata" begin
        include("auto_matrix_free_in_place_aviary.jl")
    end
    @safetestset "out-of-place" begin
        include("auto_matrix_free_out_of_place.jl")
    end
    @safetestset "out-of-place, with Aviary metadata" begin
        include("auto_matrix_free_out_of_place_aviary.jl")
    end
end

