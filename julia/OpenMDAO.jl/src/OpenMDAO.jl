module OpenMDAO

using OpenMDAOCore: OpenMDAOCore
using PythonCall: PythonCall

export om, make_component

# load python api
const om = PythonCall.pynew()
const omjlcomps = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(om, PythonCall.pyimport("openmdao.api"))
    PythonCall.pycopy!(omjlcomps, PythonCall.pyimport("omjlcomps"))
end

make_component(comp::OpenMDAOCore.AbstractExplicitComp) = omjlcomps.JuliaExplicitComp(jlcomp=comp)
make_component(comp::OpenMDAOCore.AbstractImplicitComp) = omjlcomps.JuliaImplicitComp(jlcomp=comp)

end # module
