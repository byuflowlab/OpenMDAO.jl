using Test
using ComponentArrays: ComponentVector
using JuliaExampleComponents: f_paraboloid!, f_circle!

@testset "paraboloid" begin
    x = 2.0
    y = 3.0
    X_ca = ComponentVector(x=x, y=y)
    Y_ca = ComponentVector(f_xy=0.0)

    f_paraboloid!(Y_ca, X_ca, nothing)
    @test Y_ca.f_xy ≈ (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0
end

@testset "circle" begin
    r = 2.0
    X_ca = ComponentVector(r=r)
    Y_ca = ComponentVector(area=0.0)
    f_circle!(Y_ca, X_ca, nothing)
    @test Y_ca[:area] ≈ pi*r^2
end
