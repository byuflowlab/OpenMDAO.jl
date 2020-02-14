module RequiredMethodsTests

using Test
using OpenMDAO

@testset "ExplicitComp required methods check" begin
    include("explicit_required_methods.jl")
end

@testset "ImplicitComp required methods check" begin
    include("implicit_required_methods.jl")
end

end # module
