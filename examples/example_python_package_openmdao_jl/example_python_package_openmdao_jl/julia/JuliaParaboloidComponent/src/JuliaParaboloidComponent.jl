module JuliaParaboloidComponent

using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using OpenMDAOCore: OpenMDAOCore
using ForwardDiff: ForwardDiff
using SparseMatrixColorings: SparseMatrixColorings

function f_paraboloid!(Y_ca, X_ca, params)
    # Get the inputs:
    x = @view(X_ca[:x])
    y = @view(X_ca[:y])
    # Could also do this:
    # x = X_ca.x
    # y = X_ca.y
    # or even this
    # (; x, y) = X_ca

    # Get the output:
    f_xy = @view(Y_ca[:f_xy])
    # Again, could also do this:
    # f_xy = Y_ca.f_xy
    # or
    # (; f_xy) = Y_ca

    # Do the calculation:
    @. f_xy = (x - 3.0)^2 + x*y + (y + 4.0)^2 - 3.0

    # No return value for in-place callback function.
    return nothing
end

function get_parabaloid_comp()
    X_ca = ComponentVector(x=1.0, y=1.0)
    Y_ca = ComponentVector(f_xy=0.0)

    # Absolute tolerance for detecting zero entries in the Jacobian.
    sparse_atol = 1e-10
    # Use ForwardDiff.jl to calculate the Jacobian and determine sparsity, evaluating the Jacobian multiple times with a perturbed input vector.
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    # Coloring algorithm for determining the color of the Jacobian.
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    # AD library: will also use ForwardDiff.jl to calculate the sparse jacobian each time.
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    # Create the OpenMDAOCore.jl component.
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_paraboloid!, Y_ca, X_ca; params=nothing)

    return comp
end

end # module JuliaParaboloidComponent
