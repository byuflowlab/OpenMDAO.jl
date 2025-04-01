using OpenMDAOCore: OpenMDAOCore, _convert_val
using Unitful: Unitful, uparse, ustrip, uconvert

val = 2143.0

@test _convert_val("rad/s", val, "rev/min") ≈ val*2*pi/60
@test all(_convert_val("rad/s", fill(val, 2, 3, 4), "rev/min") .≈ val*2*pi/60)

@test _convert_val("W", val, "hp") ≈ val*745.7
@test all(_convert_val("W", fill(val, 2, 3, 4), "hp") .≈ val*745.7)

@test _convert_val("mi/h", val, "m/s") ≈ val/0.0254/12/3/1760*3600
@test all(_convert_val("mi/h", fill(val, 2, 3, 4), "m/s") .≈ val/0.0254/12/3/1760*3600)
