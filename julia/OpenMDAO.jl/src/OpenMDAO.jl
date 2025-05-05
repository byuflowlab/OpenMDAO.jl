module OpenMDAO

# using Pkg: Pkg
using OpenMDAOCore: OpenMDAOCore
using PythonCall: PythonCall

export om, make_component, DymosifiedCompWrapper

# load python api
const om = PythonCall.pynew()
const omjlcomps = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(om, PythonCall.pyimport("openmdao.api"))
    # Pkg.add("OpenMDAOCore")
    PythonCall.pycopy!(omjlcomps, PythonCall.pyimport("omjlcomps"))
end

"""
    make_component(comp::OpenMDAOCore.AbstractComp)

Convinience method for creating either a `JuliaExplicitComp` or `JuliaImplicitComp`, depending on if `comp` is `<:OpenMDAOCore.AbstractExplicitComp` or `<:OpenMDAOCore.AbstractImplicitComp`, respectively.
"""
make_component
make_component(comp::OpenMDAOCore.AbstractExplicitComp) = omjlcomps.JuliaExplicitComp(jlcomp=comp)
make_component(comp::OpenMDAOCore.AbstractImplicitComp) = omjlcomps.JuliaImplicitComp(jlcomp=comp)

end # module
