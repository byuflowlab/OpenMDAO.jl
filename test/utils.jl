module UtilsTests

using Test
using OpenMDAO

@testset "get_rows_cols check" begin
    ss_sizes = Dict(:i=>2, :j=>3, :k=>4)

    of_ss = [:i]
    wrt_ss = [:i]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1])
    @test all(cols .== [0, 1])

    of_ss = [:i]
    wrt_ss = [:j]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 1, 1, 1])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:i]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 0, 0, 1, 1, 1])

    of_ss = [:i, :j]
    wrt_ss = [:j]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 1, 2, 0, 1, 2])

    of_ss = [:i, :j]
    wrt_ss = [:j, :i]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 1, 2, 3, 4, 5])
    @test all(cols .== [0, 2, 4, 1, 3, 5])

    of_ss = [:i, :j]
    wrt_ss = [:j, :k]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    ss_sizes = Dict(:k=>2, :i=>3, :j=>4)

    of_ss = [:k, :i]
    wrt_ss = [:i, :j]
    rows, cols = OpenMDAO.get_rows_cols(ss_sizes=ss_sizes, of_ss=of_ss, wrt_ss=wrt_ss)
    @test all(rows .== [0, 0, 0, 0, 1, 1, 1, 1, 2, 2,  2,  2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,  5,  5])
    @test all(cols .== [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
end

end # module
