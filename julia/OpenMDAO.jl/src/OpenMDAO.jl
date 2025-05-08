module OpenMDAO

using Pkg: Pkg
using OpenMDAOCore: OpenMDAOCore
using PythonCall: PythonCall

export om, omjlcomps, make_component

# load python api
const om = PythonCall.pynew()
const omjlcomps = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(om, PythonCall.pyimport("openmdao.api"))
    openmdao_core_uuid = Base.UUID("24d19c10-6eee-420f-95df-4537264b2753")
    openmdao_core_pkg_info = Pkg.dependencies()[openmdao_core_uuid]
    if ! openmdao_core_pkg_info.is_direct_dep
        Pkg.add("OpenMDAOCore")
    end
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
