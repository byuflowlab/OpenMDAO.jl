module OpenMDAO

export OM_Ecomp_Data, OM_Icomp_Data, OM_Options_Data, OM_Var_Data, OM_Partials_Data

struct OM_Ecomp_Data
    compute
    compute_partials
    options
    inputs
    outputs
    partials
end

struct OM_Icomp_Data
    apply_nonlinear
    linearize
    options
    inputs
    outputs
    partials
end

struct OM_Options_Data
    name
    type
    val
end

struct OM_Var_Data
    name
    shape
    val
    units
end

OM_Var_Data(name, shape, val) = OM_Var_Data(name, shape, val, "")

struct OM_Partials_Data
    of
    wrt
end

end # module
