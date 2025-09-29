function f_arctan_yox!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])
    y = @view(X[:y])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = atan(y/x)

    return nothing
end

function get_arctan_yox_comp(size)
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(g=ones(size))
    X_ca = ComponentVector(x=ones(size), y=ones(size))
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_arctan_yox!, Y_ca, X_ca)
    # arctan_yox_comp = make_component(comp)
    return comp
end

function f_circle!(Y, X, params)
    # Get views of inputs.
    r = X.r

    # Get views of outputs.
    # area = Y.area
    area = @view(Y[:area])

    # Perform the calculation.
    @. area = pi*r^2

    return nothing
end

function get_circle_comp()
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(area=1.0)
    X_ca = ComponentVector(r=1.0)
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_circle!, Y_ca, X_ca)
    return comp
end

function f_r_con!(Y, X, params)
    # Get views of inputs.
    (; x, y, r) = X

    # Get views of outputs.
    g = Y.g

    # Perform the calculation.
    @. g = x^2 + y^2 - r

    return nothing
end

function get_r_con_comp(size)
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(g=ones(size))
    X_ca = ComponentVector(x=ones(size), y=ones(size), r=1.0)
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_r_con!, Y_ca, X_ca)
    return comp
end

function f_theta_con!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])

    # Get views of outputs.
    g = @view(Y[:g])

    # Create the theta parameter.
    theta_min = params.theta_min
    theta_max = params.theta_max
    theta = range(theta_min, theta_max; length=length(x))

    # Perform the calculation.
    @. g = x - theta

    return nothing
end

function get_theta_con_comp(size)
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(g=ones(size))
    X_ca = ComponentVector(x=ones(size))
    params_theta_con = (theta_min=0.0, theta_max=pi/4)
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_theta_con!, Y_ca, X_ca; params=params_theta_con)
end

function f_delta_theta_con!(Y, X, params)
    # Get views of inputs.
    # even = @view(X[:even])
    # odd = @view(X[:odd])
    (; even, odd) = X

    # Get views of outputs.
    # g = @view(Y[:g])
    (; g) = Y

    # Perform the calculation.
    @. g = even - odd

    return nothing
end

function get_delta_theta_con_comp(size)
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(g=ones(size÷2))
    X_ca = ComponentVector(even=ones(size÷2), odd=ones(size÷2))
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_delta_theta_con!, Y_ca, X_ca)
    return comp
end

function f_l_conx!(Y, X, params)
    # Get views of inputs.
    x = @view(X[:x])

    # Get views of outputs.
    g = @view(Y[:g])

    # Perform the calculation.
    @. g = x - 1

    return nothing
end

function get_l_conx_comp(size)
    # Use the same AD as the previous example: ForwardDiff.jl, with `PerturbedDenseSparsityDetector`.
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff(); sparsity_detector=sparsity_detector, coloring_algorithm=coloring_algorithm)

    Y_ca = ComponentVector(g=ones(size))
    X_ca = ComponentVector(x=ones(size))
    comp = OpenMDAOCore.SparseADExplicitComp(ad_backend, f_l_conx!, Y_ca, X_ca)
end
