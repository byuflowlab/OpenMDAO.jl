using Test
using ComponentArrays: ComponentVector
using JuliaExampleComponents: f_paraboloid!, f_circle!, Diode, circuit_solve

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

@testset "circuit" begin
    # circuit = MyCircuit()
    R1 = 100.0  # ohms
    R2 = 10000.0  # ohms
    I_in = 0.1  # amps
    V_g = 0.0  # volts
    D1 = Diode()
    n1_V, n2_V = circuit_solve(R1, R2, I_in, V_g, D1)
    n1_V_expected = 9.90804735
    n2_V_expected = 0.71278185
    @test abs(n1_V - n1_V_expected)/n1_V_expected < 1e-8
    @test abs(n2_V - n2_V_expected)/n2_V_expected < 1e-8
end
