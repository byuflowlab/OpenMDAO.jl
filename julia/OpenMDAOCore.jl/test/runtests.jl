using OpenMDAOCore
using Test
using Documenter
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using SparseArrays: sparse, findnz, nnz, issparse
using SparseDiffTools: matrix_colors, ForwardColorJacCache

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

@testset "get_rows_cols_dict_from_sparsity" begin

    @testset "0d, 1d, 2d" begin
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
        rcdict = OpenMDAOCore.get_rows_cols_dict_from_sparsity(J_ca_sparse)

        # Shouldn't make a difference if the Jacobian is sparse or not.
        rcdict_not_sparse_J_ca = OpenMDAOCore.get_rows_cols_dict_from_sparsity(J_ca)
        @test rcdict == rcdict_not_sparse_J_ca

        # Short function that puts the rows and cols in a standard order for comparison purposes.
        function rows_cols_normalize(rows, cols)
            out = sortslices(hcat(rows, cols); dims=1)
            return out[:, 1], out[:, 2]
        end

        ss_sizes = Dict(:i=>N, :j=>M, :s=>1)

        rows, cols = rows_cols_normalize(rcdict[:d, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:d, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:d, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:d, :crev]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :crev]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :crev]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:frev, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:frev, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:frev, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:frev, :crev]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j, :i], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:s], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :crev]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:s], wrt_ss=[:j, :i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rcdict[:h, :a]
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
    end

    @testset "3d" begin
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
        rcdict = OpenMDAOCore.get_rows_cols_dict_from_sparsity(J_ca_sparse)

        rcdict_not_sparse_J_ca = OpenMDAOCore.get_rows_cols_dict_from_sparsity(J_ca)
        @test rcdict == rcdict_not_sparse_J_ca

        # Short function that puts the rows and cols in a standard order for comparison purposes.
        function rows_cols_normalize(rows, cols)
            out = sortslices(hcat(rows, cols); dims=1)
            return out[:, 1], out[:, 2]
        end

        ss_sizes = Dict(:i=>I, :j=>J, :k=>K)

        rows, cols = rows_cols_normalize(rcdict[:d, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:d, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:d, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:e, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:j], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:f, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :a]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:i], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :b]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:i, :j], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)

        rows, cols = rows_cols_normalize(rcdict[:g, :c]...)
        rows_check, cols_check = rows_cols_normalize(OpenMDAOCore.get_rows_cols(; ss_sizes=ss_sizes, of_ss=[:k, :j, :i], wrt_ss=[:j, :i, :k], column_major=false, zero_based_indexing=false)...)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
    end

end

@testset "AbstractAutoSparseForwardDiffExplicitComp" begin

    @testset "manual sparsity" begin
        struct Comp1{TCompute,TX,TY,TJ,TCache,TRCDict} <: AbstractAutoSparseForwardDiffExplicitComp where {TCompute,TX,TY,TJ,TCache,TRCDict}
            compute_forwarddiffable!::TCompute
            X_ca::TX
            Y_ca::TY
            J_ca_sparse::TJ
            jac_cache::TCache
            rcdict::TRCDict
        end

        function Comp1(M, N)
            compute_forwarddiffable! = let M=M, N=N
                (Y, X)->begin
                    a = only(X[:a])
                    b = @view X[:b]
                    c = @view X[:c]
                    d = @view X[:d]
                    e = @view Y[:e]
                    f = @view Y[:f]
                    g = @view Y[:g]

                    for n in 1:N
                        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
                        for m in 1:M
                            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
                            g[n, m] = 10*sin(b[n])*cos(d[m, n])
                        end
                    end
                    return nothing
                end
            end
            X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
            Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

            # Create a dense ComponentMatrix from the input and output arrays.
            J_ca = Y_ca.*X_ca'

            # Define the sparsity by writing ones and zeros to the J_ca dense `ComponentMatrix`.
            J_ca .= 0.0
            for n in 1:N
                @view(J_ca[:e, :a])[n] = 1.0
                @view(J_ca[:e, :b])[n, n] = 1.0
                for m in 1:M
                    @view(J_ca[:e, :c])[n, m] = 1.0
                    @view(J_ca[:e, :d])[n, m, n] = 1.0

                    @view(J_ca[:f, :a])[m, n] = 1.0
                    @view(J_ca[:f, :b])[m, n, n] = 1.0
                    @view(J_ca[:f, :c])[m, n, m] = 1.0
                    @view(J_ca[:f, :d])[m, n, m, n] = 1.0

                    @view(J_ca[:g, :b])[n, m, n] = 1.0
                    @view(J_ca[:g, :d])[n, m, m, n] = 1.0
                end
            end

            # Create a sparse matrix version of J_ca.
            J_ca_sparse = ComponentMatrix(sparse(getdata(J_ca)), getaxes(J_ca))

            # Get a dictionary describing the non-zero rows and cols for each subjacobian.
            rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

            # Get some colors!
            colors = matrix_colors(getdata(J_ca_sparse))

            # Create the cache object for `forwarddiff_color_jacobian!`.
            jac_cache = ForwardColorJacCache(compute_forwarddiffable!, X_ca; dx=Y_ca, colorvec=colors, sparsity=getdata(J_ca_sparse))

            return Comp1(compute_forwarddiffable!, X_ca, Y_ca, J_ca_sparse, jac_cache, rcdict)
        end

        # Don't worry about units for now.
        get_units(self::Comp1, varname) = nothing

        # Create the component.
        N = 3
        M = 4
        comp = Comp1(M, N)

        rcdict = get_rows_cols_dict(comp)

        inputs_dict = ca2strdict(get_input_ca(comp))
        inputs_dict["a"] = 2.0
        inputs_dict["b"] .= range(3.0, 4.0; length=N)
        inputs_dict["c"] .= range(5.0, 6.0; length=M)
        inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
        outputs_dict = ca2strdict(get_output_ca(comp))

        OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
        a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
        e_check = 2*a^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
        @test all(outputs_dict["e"] .≈ e_check)

        f_check = 6*a^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
        @test all(outputs_dict["f"] .≈ f_check)

        g_check = 10 .* sin.(b).*cos.(transpose(d))
        @test all(outputs_dict["g"] .≈ g_check)

        J_ca_sparse = get_sparse_jacobian_ca(comp)
        @test issparse(getdata(J_ca_sparse))
        @test size(getdata(J_ca_sparse)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
        @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
        partials_dict = rcdict2strdict(rcdict)
        OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

        rows, cols = rcdict[:e, :a]
        vals = partials_dict["e", "a"]
        @test size(vals) == (N,)
        a = inputs_dict["a"]
        deda_check = zeros(N)
        for n in 1:N
            deda_check[n] = 4*a
        end
        deda_check_sparse = sparse(reshape(deda_check, N))
        # `e` is a vector of length `N` and `a` is scalar, so the Jacobian isn't actually a Matrix (and isn't really sparse).
        rows_check, vals_check = findnz(deda_check_sparse)
        cols_check = fill(1, N)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :b]
        vals = partials_dict["e", "b"]
        @test size(vals) == (N,)
        b = inputs_dict["b"]
        dedb_check = zeros(N, N)
        for n in 1:N
            dedb_check[n, n] = (3*2.1)*b[n]^1.1
        end
        dedb_check_sparse = sparse(reshape(dedb_check, N, N))
        rows_check, cols_check, vals_check = findnz(dedb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :c]
        vals = partials_dict["e", "c"]
        @test size(vals) == (N*M,)
        c = inputs_dict["c"]
        dedc_check = zeros(N, M)
        for m in 1:M
            for n in 1:N
                dedc_check[n, m] = (4*2.2)*c[m]^1.2
            end
        end
        dedc_check_sparse = sparse(reshape(dedc_check, N, M))
        rows_check, cols_check, vals_check = findnz(dedc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :d]
        vals = partials_dict["e", "d"]
        @test size(vals) == (M*N,)
        d = inputs_dict["d"]
        dedd_check = zeros(N, M, N)
        for n in 1:N
            for m in 1:M
                dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
            end
        end
        dedd_check_sparse = sparse(reshape(dedd_check, N, M*N))
        rows_check, cols_check, vals_check = findnz(dedd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :a]
        vals = partials_dict["f", "a"]
        @test size(vals) == (M*N,)
        a = inputs_dict["a"]
        dfda_check = zeros(M, N)
        for m in 1:M
            for n in 1:N
                dfda_check[m, n] = (6*2.4)*a^1.4
            end
        end
        dfda_check_sparse = sparse(reshape(dfda_check, M*N, 1))
        rows_check, cols_check, vals_check = findnz(dfda_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :b]
        vals = partials_dict["f", "b"]
        @test size(vals) == (M*N,)
        dfdb_check = zeros(M, N, N)
        for n in 1:N
            for m in 1:M
                dfdb_check[m, n, n] = (7*2.5)*b[n]^1.5
            end
        end
        dfdb_check_sparse = sparse(reshape(dfdb_check, M*N, N))
        rows_check, cols_check, vals_check = findnz(dfdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :c]
        vals = partials_dict["f", "c"]
        @test size(vals) == (M*N,)
        dfdc_check = zeros(M, N, M)
        for n in 1:N
            for m in 1:M
                dfdc_check[m, n, m] = (8*2.6)*c[m]^1.6
            end
        end
        dfdc_check_sparse = sparse(reshape(dfdc_check, M*N, M))
        rows_check, cols_check, vals_check = findnz(dfdc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :d]
        vals = partials_dict["f", "d"]
        @test size(vals) == (M*N,)
        dfdd_check = zeros(M, N, M, N)
        for n in 1:N
            for m in 1:M
                dfdd_check[m, n, m, n] = (9*2.7)*d[m, n]^1.7
            end
        end
        dfdd_check_sparse = sparse(reshape(dfdd_check, M*N, M*N))
        rows_check, cols_check, vals_check = findnz(dfdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:g, :a]
        vals = partials_dict["g", "a"]
        @test size(vals) == (0,)
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64

        rows, cols = rcdict[:g, :b]
        vals = partials_dict["g", "b"]
        @test size(vals) == (N*M,)
        dgdb_check = zeros(N, M, N)
        for m in 1:M
            for n in 1:N
                dgdb_check[n, m, n] = 10*cos(b[n])*cos(d[m, n])
            end
        end
        dgdb_check_sparse = sparse(reshape(dgdb_check, N*M, N))
        rows_check, cols_check, vals_check = findnz(dgdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:g, :c]
        vals = partials_dict["g", "c"]
        @test size(vals) == (0,)
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64

        rows, cols = rcdict[:g, :d]
        vals = partials_dict["g", "d"]
        @test size(vals) == (N*M,)
        dgdd_check = zeros(N, M, M, N)
        for m in 1:M
            for n in 1:N
                dgdd_check[n, m, m, n] = -10*sin(b[n])*sin(d[m, n])
            end
        end
        dgdd_check_sparse = sparse(reshape(dgdd_check, N*M, M*N))
        rows_check, cols_check, vals_check = findnz(dgdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)
    end

    @testset "automatic sparsity" begin

        struct Comp2{TCompute,TX,TY,TJ,TCache,TRCDict} <: AbstractAutoSparseForwardDiffExplicitComp where {TCompute,TX,TY,TJ,TCache,TRCDict}
            compute_forwarddiffable!::TCompute
            X_ca::TX
            Y_ca::TY
            J_ca_sparse::TJ
            jac_cache::TCache
            rcdict::TRCDict
        end

        function Comp2(M, N)
            compute_forwarddiffable! = let M=M, N=N
                (Y, X)->begin
                    a = only(X[:a])
                    b = @view X[:b]
                    c = @view X[:c]
                    d = @view X[:d]
                    e = @view Y[:e]
                    f = @view Y[:f]
                    g = @view Y[:g]

                    for n in 1:N
                        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
                        for m in 1:M
                            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
                            g[n, m] = 10*sin(b[n])*cos(d[m, n])
                        end
                    end
                    return nothing
                end
            end
            X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
            @view(X_ca[:a]) .= 2.0
            @view(X_ca[:b]) .= range(3.0, 4.0; length=N)
            @view(X_ca[:c]) .= range(5.0, 6.0; length=M)
            @view(X_ca[:d]) .= reshape(range(7.0, 8.0; length=M*N), M, N)
            Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

            # Create a dense ComponentMatrix from the input and output arrays.
            J_ca = Y_ca.*X_ca'

            # Hopefully create the sparsity automatically.
            generate_perturbed_jacobian!(J_ca, compute_forwarddiffable!, Y_ca, X_ca)

            # Create a sparse matrix version of J_ca.
            J_ca_sparse = ComponentMatrix(sparse(getdata(J_ca)), getaxes(J_ca))

            # Get a dictionary describing the non-zero rows and cols for each subjacobian.
            rcdict = get_rows_cols_dict_from_sparsity(J_ca_sparse)

            # Get some colors!
            colors = matrix_colors(getdata(J_ca_sparse))

            # Create the cache object for `forwarddiff_color_jacobian!`.
            jac_cache = ForwardColorJacCache(compute_forwarddiffable!, X_ca; dx=Y_ca, colorvec=colors, sparsity=getdata(J_ca_sparse))

            return Comp2(compute_forwarddiffable!, X_ca, Y_ca, J_ca_sparse, jac_cache, rcdict)
        end

        # Don't worry about units for now.
        get_units(self::Comp2, varname) = nothing

        # Create the component.
        N = 3
        M = 4
        comp = Comp2(M, N)

        rcdict = get_rows_cols_dict(comp)

        inputs_dict = ca2strdict(get_input_ca(comp))
        inputs_dict["a"] = 2.0
        inputs_dict["b"] .= range(3.0, 4.0; length=N)
        inputs_dict["c"] .= range(5.0, 6.0; length=M)
        inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
        outputs_dict = ca2strdict(get_output_ca(comp))

        OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
        a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
        e_check = 2*a^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
        @test all(outputs_dict["e"] .≈ e_check)

        f_check = 6*a^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
        @test all(outputs_dict["f"] .≈ f_check)

        g_check = 10 .* sin.(b).*cos.(transpose(d))
        @test all(outputs_dict["g"] .≈ g_check)

        J_ca_sparse = get_sparse_jacobian_ca(comp)
        @test issparse(getdata(J_ca_sparse))
        @test size(getdata(J_ca_sparse)) == (length(get_output_ca(comp)), length(get_input_ca(comp)))
        @test nnz(getdata(J_ca_sparse)) == N + N + N*M + N*M + M*N + M*N + M*N + M*N + N*M + N*M
        partials_dict = rcdict2strdict(rcdict)
        OpenMDAOCore.compute_partials!(comp, inputs_dict, partials_dict)

        rows, cols = rcdict[:e, :a]
        vals = partials_dict["e", "a"]
        @test size(vals) == (N,)
        a = inputs_dict["a"]
        deda_check = zeros(N)
        for n in 1:N
            deda_check[n] = 4*a
        end
        deda_check_sparse = sparse(reshape(deda_check, N))
        # `e` is a vector of length `N` and `a` is scalar, so the Jacobian isn't actually a Matrix (and isn't really sparse).
        rows_check, vals_check = findnz(deda_check_sparse)
        cols_check = fill(1, N)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :b]
        vals = partials_dict["e", "b"]
        @test size(vals) == (N,)
        b = inputs_dict["b"]
        dedb_check = zeros(N, N)
        for n in 1:N
            dedb_check[n, n] = (3*2.1)*b[n]^1.1
        end
        dedb_check_sparse = sparse(reshape(dedb_check, N, N))
        rows_check, cols_check, vals_check = findnz(dedb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :c]
        vals = partials_dict["e", "c"]
        @test size(vals) == (N*M,)
        c = inputs_dict["c"]
        dedc_check = zeros(N, M)
        for m in 1:M
            for n in 1:N
                dedc_check[n, m] = (4*2.2)*c[m]^1.2
            end
        end
        dedc_check_sparse = sparse(reshape(dedc_check, N, M))
        rows_check, cols_check, vals_check = findnz(dedc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:e, :d]
        vals = partials_dict["e", "d"]
        @test size(vals) == (M*N,)
        d = inputs_dict["d"]
        dedd_check = zeros(N, M, N)
        for n in 1:N
            for m in 1:M
                dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
            end
        end
        dedd_check_sparse = sparse(reshape(dedd_check, N, M*N))
        rows_check, cols_check, vals_check = findnz(dedd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :a]
        vals = partials_dict["f", "a"]
        @test size(vals) == (M*N,)
        a = inputs_dict["a"]
        dfda_check = zeros(M, N)
        for m in 1:M
            for n in 1:N
                dfda_check[m, n] = (6*2.4)*a^1.4
            end
        end
        dfda_check_sparse = sparse(reshape(dfda_check, M*N, 1))
        rows_check, cols_check, vals_check = findnz(dfda_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :b]
        vals = partials_dict["f", "b"]
        @test size(vals) == (M*N,)
        dfdb_check = zeros(M, N, N)
        for n in 1:N
            for m in 1:M
                dfdb_check[m, n, n] = (7*2.5)*b[n]^1.5
            end
        end
        dfdb_check_sparse = sparse(reshape(dfdb_check, M*N, N))
        rows_check, cols_check, vals_check = findnz(dfdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :c]
        vals = partials_dict["f", "c"]
        @test size(vals) == (M*N,)
        dfdc_check = zeros(M, N, M)
        for n in 1:N
            for m in 1:M
                dfdc_check[m, n, m] = (8*2.6)*c[m]^1.6
            end
        end
        dfdc_check_sparse = sparse(reshape(dfdc_check, M*N, M))
        rows_check, cols_check, vals_check = findnz(dfdc_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:f, :d]
        vals = partials_dict["f", "d"]
        @test size(vals) == (M*N,)
        dfdd_check = zeros(M, N, M, N)
        for n in 1:N
            for m in 1:M
                dfdd_check[m, n, m, n] = (9*2.7)*d[m, n]^1.7
            end
        end
        dfdd_check_sparse = sparse(reshape(dfdd_check, M*N, M*N))
        rows_check, cols_check, vals_check = findnz(dfdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:g, :a]
        vals = partials_dict["g", "a"]
        @test size(vals) == (0,)
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64

        rows, cols = rcdict[:g, :b]
        vals = partials_dict["g", "b"]
        @test size(vals) == (N*M,)
        dgdb_check = zeros(N, M, N)
        for m in 1:M
            for n in 1:N
                dgdb_check[n, m, n] = 10*cos(b[n])*cos(d[m, n])
            end
        end
        dgdb_check_sparse = sparse(reshape(dgdb_check, N*M, N))
        rows_check, cols_check, vals_check = findnz(dgdb_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)

        rows, cols = rcdict[:g, :c]
        vals = partials_dict["g", "c"]
        @test size(vals) == (0,)
        @test rows == Vector{Int}()
        @test cols == Vector{Int}()
        @test eltype(vals) == Float64

        rows, cols = rcdict[:g, :d]
        vals = partials_dict["g", "d"]
        @test size(vals) == (N*M,)
        dgdd_check = zeros(N, M, M, N)
        for m in 1:M
            for n in 1:N
                dgdd_check[n, m, m, n] = -10*sin(b[n])*sin(d[m, n])
            end
        end
        dgdd_check_sparse = sparse(reshape(dgdd_check, N*M, M*N))
        rows_check, cols_check, vals_check = findnz(dgdd_check_sparse)
        @test all(rows .== rows_check)
        @test all(cols .== cols_check)
        @test all(vals .≈ vals_check)
    end
end
