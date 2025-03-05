using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using ForwardDiff: ForwardDiff
using SparseMatrixColorings: SparseMatrixColorings
using OpenMDAOCore: OpenMDAOCore

function f_radius(X, params)
    return ComponentVector(Rtip=0.5*X.Dtip, thrust=3*X.Dtip^2)
end
    
function f_radius!(Y, X, params)
    Y.Rtip = 0.5*X.Dtip
    Y.thrust = 3*X.Dtip^2
    return nothing
end
    

function get_aviary_matrix_free_test_comp(; in_place, aviary_input_vars, aviary_output_vars, aviary_meta_data)
    ad_backend = ADTypes.AutoForwardDiff()

    units_dict = Dict(:Rtip=>"ft")

    X_ca = ComponentVector{Float64}()
    if in_place
        Y_ca = ComponentVector{Float64}(Rtip=0.0)
        comp = OpenMDAOCore.MatrixFreeADExplicitComp(ad_backend, f_radius!, Y_ca, X_ca; aviary_input_vars, aviary_output_vars, aviary_meta_data, units_dict)
    else
        comp = OpenMDAOCore.MatrixFreeADExplicitComp(ad_backend, f_radius, X_ca; aviary_input_vars, aviary_output_vars, aviary_meta_data, units_dict)
    end

    return comp
end

function get_aviary_sparse_test_comp(; in_place, aviary_input_vars, aviary_output_vars, aviary_meta_data)
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    units_dict = Dict(:Rtip=>"ft")

    X_ca = ComponentVector{Float64}()
    if in_place
        Y_ca = ComponentVector{Float64}(Rtip=0.0)
        comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_radius!, Y_ca, X_ca; aviary_input_vars, aviary_output_vars, aviary_meta_data, units_dict)
    else
        comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_radius, X_ca; aviary_input_vars, aviary_output_vars, aviary_meta_data, units_dict)
    end

    return comp
end
