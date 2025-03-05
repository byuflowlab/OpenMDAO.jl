using OpenMDAOCore: OpenMDAOCore, _unitfulify_units
using Unitful: Unitful, uparse, ustrip, uconvert
using UnitfulAngles: UnitfulAngles  # needed for converting an OpenMDAO `rev` to a UnitfulAngles `turn`

iu = uparse(_unitfulify_units("rad/s"); unit_context=[Unitful, UnitfulAngles, OpenMDAOCore])
du = uparse(_unitfulify_units("rev/min"); unit_context=[Unitful, UnitfulAngles, OpenMDAOCore])
val = 2000.0
val_convert = val * ustrip(uconvert(iu, one(Float64)*du))
@test val_convert ≈ val*2*pi/60

iu = uparse(_unitfulify_units("hp"); unit_context=[Unitful, UnitfulAngles, OpenMDAOCore])
du = uparse(_unitfulify_units("W"); unit_context=[Unitful, UnitfulAngles, OpenMDAOCore])
val = 2000.0
val_convert = val * ustrip(uconvert(iu, one(Float64)*du))
@test val_convert ≈ 2000.0/745.7
