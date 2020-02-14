module OptionalMethodsTests

using Test
using OpenMDAO

@testset "ExplicitComp optional methods check" begin
    include("explicit_optional_methods.jl")
end

@testset "ImplicitComp optional methods check" begin
    include("implicit_optional_methods.jl")
end

end # module
