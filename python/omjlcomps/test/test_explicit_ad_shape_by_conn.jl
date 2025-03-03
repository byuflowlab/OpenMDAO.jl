using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using ForwardDiff: ForwardDiff
using SparseMatrixColorings: SparseMatrixColorings
using OpenMDAOCore: OpenMDAOCore

my_reverse(x::Number) = x
my_reverse(x) = reverse(x)

my_sum(x::Number; kwargs...) = x
my_sum(x; kwargs...) = sum(x; kwargs...)

function f1(X, params)
    return ComponentVector(y=0.5.*X.x, z=3.0.*my_reverse(X.x).*X.x.^2 .+ 4.0.*my_sum(X.x; dims=ndims(X.x)))
end
    
function f1!(Y, X, params)
    Y.y .= 0.5.*X.x
    Y.z .= 3.0.*my_reverse(X.x).*X.x.^2 .+ 4.0.*my_sum(X.x; dims=ndims(X.x))
    return nothing
end
    

function get_matrix_free_test_comp(; in_place)
    ad_backend = ADTypes.AutoForwardDiff()

    X_ca = ComponentVector{Float64}(x=3.0)
    shape_by_conn_dict = Dict(:x=>true)
    copy_shape_dict = Dict(:y=>:x, :z=>:x)
    if in_place
        Y_ca = ComponentVector{Float64}(y=0.0, z=0.0)
        comp = OpenMDAOCore.MatrixFreeADExplicitComp(ad_backend, f1!, Y_ca, X_ca; shape_by_conn_dict, copy_shape_dict)
    else
        comp = OpenMDAOCore.MatrixFreeADExplicitComp(ad_backend, f1, X_ca; shape_by_conn_dict, copy_shape_dict)
    end

    return comp
end

function get_sparse_test_comp(; in_place)
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    X_ca = ComponentVector{Float64}(x=3.0)
    shape_by_conn_dict = Dict(:x=>true)
    copy_shape_dict = Dict(:y=>:x, :z=>:x)
    if in_place
        Y_ca = ComponentVector{Float64}(y=0.0, z=0.0)
        comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f1!, Y_ca, X_ca; shape_by_conn_dict, copy_shape_dict)
    else
        comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f1, X_ca; shape_by_conn_dict, copy_shape_dict)
    end

    return comp
end
