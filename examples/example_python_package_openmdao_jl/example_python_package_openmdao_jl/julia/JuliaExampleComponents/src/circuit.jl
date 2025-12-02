struct Resistor{TF}
    R::TF
end

function current(resistor::Resistor, V_in, V_out)
    I = (V_in - V_out)/resistor.R
    return I
end

Resistor(; R=1.0) = Resistor(R)

struct Diode
    Is::Float64
    Vt::Float64
end

Diode(; Is=1e-15, Vt=0.025875) = Diode(Is, Vt)

function current(diode::Diode, V_in, V_out)
    deltaV = V_in - V_out
	Is = diode.Is
	Vt = diode.Vt
    I = Is * (exp(deltaV / Vt) - 1)
    return I
end

# struct Node
#     n_in::Int
# 	n_out::Int
# end

# # Default to 1 input and output current.
# Node(; n_in=1, n_out=1) = Node(n_in, n_out)

# function residual_current(node::Node, I_in, I_out)
#     return sum(I_in) - sum(I_out)
# end

struct MyCircuit{TR1,TR2,TD1}
    # n1::Node
    # n2::Node
    R1::TR1
    R2::TR2
    D1::TD1
end

function MyCircuit()
    # n1 = Node(1, 2)
    # n2 = Node()
    R1 = Resistor(100.0)
    R2 = Resistor(10000.0)
    D1 = Diode()
    # return MyCircuit(n1, n2, R1, R2, D1)
    return MyCircuit(R1, R2, D1)
end

# Now, what am I solving for?
# I'm balancing the current at the nodes.
# So, hmm...
function circuit_residual(circuit::MyCircuit, n1_V, n2_V, I_in, V_g)
    # OK, how does this work?
    # I need the currents at each node to sum to zero.
    # So that's the residual I'm trying to get.
    # So, how do I calculate that?

    # Node 1 has 1 input, two outputs.
    # On the group level, what are the knowns?
    #
    #   * We know the current source is 0.1.
    #   * We know the ground voltage is 0.0.

    # I_in = circuit.I_source
    # V_g = circuit.V_ground

    # # So, the outputs are:
    # n1_V = x.n1_V
    # n2_V = x.n2_V

    # Now I can find the current through R1.
    R1_I = current(circuit.R1, n1_V, V_g)

    # I can also get the current through R2.
    R2_I = current(circuit.R2, n1_V, n2_V)

    # And the current through D1.
    D1_I = current(circuit.D1, n2_V, V_g)

    # Now I can get the residuals for each node.
    # n1_residual = residual_current(circuit.n1, [I_in], [R1_I, R2_I])
    # n2_residual = residual_current(circuit.n2, [R2_I], [D1_I])
    n1_residual = I_in - R1_I - R2_I
    n2_residual = R2_I - D1_I

    # Now return a residual vector.
    return n1_residual, n2_residual
end

function circuit_residual(y, x, p)
    # circuit = p.circuit
    D1 = p.D1
    R1 = x.R1
    R2 = x.R2
    I_in = x.I_in
    V_g = x.V_g
    n1_V = y.n1_V
    n2_V = y.n2_V

    circuit = MyCircuit(Resistor(R1), Resistor(R2), D1)
    
    n1_residual, n2_residual = circuit_residual(circuit, n1_V, n2_V, I_in, V_g)
    return ComponentVector(n1_V=n1_residual, n2_V=n2_residual)
end

function circuit_solve(x, p)
    residual_nlsolve = let x=x, p=p
        (y,)->circuit_residual(y, x, p)
    end

    TF = eltype(x)
    y = ComponentVector{TF}(n1_V=10*one(TF), n2_V=(1e-3)*one(TF))
    res = nlsolve(residual_nlsolve, y; autodiff=:forward, method=:newton, linesearch=BackTracking(), show_trace=true)
    y .= res.zero
    return y
end

function circuit_solve(R1, R2, I_in, V_g, D1::Diode)
    x = ComponentVector(R1=R1, R2=R2, I_in=I_in, V_g=V_g)
    p = (; D1=D1)
    # y = circuit_solve(x, p)
    y = implicit(circuit_solve, circuit_residual, x, p)
    n1_V = y.n1_V
    n2_V = y.n2_V
    return n1_V, n2_V
end

function f_circuit(x, p)
    Is = p.Is
    Vt = p.Vt
    R1 = x.R1
    R2 = x.R2
    # circuit = MyCircuit(Node(1, 2), Node(), R1, R2, Diode(Is, Vt))
    I_in = x.I_in
    V_g = x.V_g
    n1_V, n2_V = circuit_solve(R1, R2, I_in, V_g, Diode(Is, Vt))
    return ComponentVector(n1_V=n1_V, n2_V=n2_V)
end

function get_circuit_comp(Is, Vt)
    X_ca = ComponentVector(R1=1.0, R2=1.0, I_in=1.0, V_g=1.0)
    ad_backend = ADTypes.AutoForwardDiff()
    params = (; Is, Vt)
    units_dict = Dict(:R1=>"ohm", :R2=>"ohm", :I_in=>"A", :V_g=>"V", :n1_V=>"V", :n2_V=>"V")
    comp = OpenMDAOCore.DenseADExplicitComp(ad_backend, f_circuit, X_ca; params=params, units_dict=units_dict)
    return comp
end
