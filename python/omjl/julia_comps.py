import openmdao.api as om


class JuliaExplicitComp(om.ExplicitComponent):

    def initialize(self):
        print("initialize")
        self.options.declare('julia_comp_data')

    def setup(self):
        print("setup")
        comp_data = self.options['julia_comp_data']
        input_data = comp_data.inputs
        output_data = comp_data.outputs

        for var in input_data:
            print(f"input var {var}")
            self.add_input(var.name,
                           shape=var.shape,
                           val=var.val)

        for var in output_data:
            print(f"output var {var}")
            self.add_output(var.name, shape=var.shape, val=var.val)

        for data in comp_data.partials:
            self.declare_partials(data.of, data.wrt)

    def compute(self, inputs, outputs):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)
        # outputs_dict = comp_data.compute(inputs_dict)
        comp_data.compute(inputs_dict, outputs_dict)
        for k, v in outputs_dict.items():
            outputs[k] = v

    def compute_partials(self, inputs, partials):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        partials_dict = {}
        for part_names in comp_data.partials:
            of_wrt = part_names.of, part_names.wrt
            print(of_wrt)
            partials_dict[of_wrt] = partials[of_wrt]
        comp_data.compute_partials(inputs_dict, partials_dict)
