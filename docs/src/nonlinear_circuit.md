```@meta
CurrentModule = OpenMDAODocs
```
# A More Complicated Example: Nonlinear Circuit

```@example circuit
using OpenMDAOCore: OpenMDAOCore

struct Resistor <: OpenMDAOCore.AbstractExplicitComp
    R::Float64
end

# Default value for R.
Resistor(; R=1.0) = Resistor(R)

function OpenMDAOCore.setup(self::Resistor)
    input_data = [OpenMDAOCore.VarData("V_in"; units="V"), OpenMDAOCore.VarData("V_out"; units="V")]
    output_data = [OpenMDAOCore.VarData("I"; units="A")]

    R = self.R
    partials_data = [OpenMDAOCore.PartialsData("I", "V_in", val=1/R),
                     OpenMDAOCore.PartialsData("I", "V_out", val=-1/R),]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::Resistor, inputs, outputs)
    deltaV = inputs["V_in"][1] - inputs["V_out"][1]
    outputs["I"][1] = deltaV/self.R

    return nothing
end

struct Diode <: OpenMDAOCore.AbstractExplicitComp
    Is::Float64
    Vt::Float64
end

# Use Julia's keyword arguments to set default values.
Diode(; Is=1e-15, Vt=0.025875) = Diode(Is, Vt)

function OpenMDAOCore.setup(self::Diode)
    input_data = [OpenMDAOCore.VarData("V_in"; units="V"), OpenMDAOCore.VarData("V_out"; units="V")]
    output_data = [OpenMDAOCore.VarData("I"; units="A")]

    partials_data = [OpenMDAOCore.PartialsData("I", "V_in"), OpenMDAOCore.PartialsData("I", "V_out")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::Diode, inputs, outputs)
    deltaV = inputs["V_in"][1] - inputs["V_out"][1]
	Is = self.Is
	Vt = self.Vt
    outputs["I"][1] = Is * (exp(deltaV / Vt) - 1)
    return nothing
end

function OpenMDAOCore.compute_partials!(self::Diode, inputs, J)
	deltaV = inputs["V_in"][1] - inputs["V_out"][1]
	Is = self.Is
	Vt = self.Vt
	I = Is * exp(deltaV / Vt)

	J["I", "V_in"][1, 1] = I/Vt
	J["I", "V_out"][1, 1] = -I/Vt

	return nothing
end

struct Node <: OpenMDAOCore.AbstractImplicitComp
    n_in::Int
	n_out::Int
end

Node(; n_in=1, n_out=1) = Node(n_in, n_out)

function OpenMDAOCore.setup(self::Node)
    output_data = [OpenMDAOCore.VarData("V", val=5.0, units="V")]

    input_data = Vector{OpenMDAOCore.VarData}()
    partials_data = Vector{OpenMDAOCore.PartialsData}()

	for i in 0:self.n_in-1
        i_name = "I_in:$i"
        push!(input_data, OpenMDAOCore.VarData(i_name; units="A"))
        push!(partials_data, OpenMDAOCore.PartialsData("V", i_name; val=1.0))
	end

	for i in 0:self.n_out-1
        i_name = "I_out:$i"
        push!(input_data, OpenMDAOCore.VarData(i_name; units="A"))
        push!(partials_data, OpenMDAOCore.PartialsData("V", i_name; val=-1.0))
	end

	return input_data, output_data, partials_data
end

function OpenMDAOCore.apply_nonlinear!(self::Node, inputs, outputs, residuals)
    residuals["V"][1] = 0.0
    for i_conn in 0:self.n_in-1
        residuals["V"][1] += inputs["I_in:$i_conn"][1]
    end
    for i_conn in 0:self.n_out-1
        residuals["V"][1] -= inputs["I_out:$i_conn"][1]
    end

    return nothing
end
```

Now the run script:

```@example circuit
using OpenMDAO: om, make_component

circuit = om.Group()

circuit.add_subsystem("n1", make_component(Node(n_in=1, n_out=2)), promotes_inputs=[("I_in:0", "I_in")])
circuit.add_subsystem("n2", make_component(Node()))

circuit.add_subsystem("R1", make_component(Resistor(R=100.0)), promotes_inputs=[("V_out", "Vg")])
circuit.add_subsystem("R2", make_component(Resistor(R=10000.0)))
```
