module OpenMDAO

export OM_Ecomp_Data, OM_Var_Data, OM_Partials_Data

struct OM_Ecomp_Data
    compute
    compute_partials
    inputs
    outputs
    partials
end

struct OM_Var_Data
    name
    shape
    val
    units
end

struct OM_Partials_Data
    of
    wrt
end

OM_Var_Data(name, shape, val) = OM_Var_Data(name, shape, val, "")

end # module
