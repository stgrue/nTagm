import yaml

class TruthValueException(Exception):
    def __init__(self, faultyString):
        super(TruthValueException, self).__init__("'{}' is not a valid truth value literal.".format(faultyString))

def get_truth_value(s):
    s = s.lower()
    if s == "true" or s == "yes":
        return True
    elif s == "false" or s == "no":
        return False
    else:
        raise TruthValueException(s)

def is_list(d):
    '''Checks if the keys of dictionary d can be interpreted as indices of a list'''
    return all(key in range(len(d)) for key in d) and all(index in d for index in range(len(d)))



class Config():
    def __init__(self, yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            options_dict = yaml.load(f.read())

        self.model = Model(yaml_path.split(".")[0], options_dict["model"])
        self.training_data = DataSources(options_dict["training_data"], self.model)
        self.development_data = DataSources(options_dict["development_data"], self.model)
        self.test_data = DataSources(options_dict["test_data"], self.model)     


class Model():
    def __init__(self, name, params):
        self.name = name
        
        if not is_list(params["inputs"]):
            raise Exception("inputs of model must be a list")
        if not is_list(params["outputs"]):
            raise Exception("outputs of model must be a list")

        num_inputs = len(params["inputs"])
        self.inputs = list()
        num_outputs = len(params["outputs"])
        self.outputs = list() 

        for i in range(num_inputs):
            self.inputs.append(Input(params["inputs"][i]))
        for i in range(num_outputs):
            self.outputs.append(Output(params["outputs"][i]))
                                
        self.rnn = RNNParameters(params["rnn"])
        self.optimizer = OptimizerParameters(params["optimizer"])


class Input():
    def __init__(self, params):
        if isinstance(params, str): # Short version: Only vocabulary file specified
            self.type = "vocabulary"
            self.vocab_file_path = params
            self.learn_lookups = False
            self.additional_labels = []
            
        elif isinstance(params, dict): # Long version: Specifiy details (vocabulary vs. embeddings, etc.)
            self.type = params["type"]
            if not(self.type == "embeddings" or self.type == "vocabulary"): # Can implement raw input later
                raise Exception(self.type + " is not a valid input type")
            if self.type == "vocabulary":
                self.vocab_file_path = params["vocabulary_file"]
            if self.type == "embeddings":
                self.embeds_file_path = params["embeddings_file"]

            if "additional_labels" in params:
                self.additional_labels = params["additional_labels"]
                if not isinstance(self.additional_labels, list):
                    raise Exception("additional_labels must be a list")          
            else:
                self.additional_labels = []

            if "learn_lookups" in params:
                self.learn_lookups = params["learn_lookups"]
                if not isinstance(self.learn_lookups, bool):
                    raise Exception("learn_lookups must be a truth value")
            else:
                self.learn_lookups = False
                
        else:
            raise Exception("Input must be specified as str or dict")


class Output():
    def __init__(self, params):
        if isinstance(params, str):
            self.vocab_file_path = params
        else:
            raise Exception("Outputs must be given as vocabulary file paths") # Can implement raw output later


class RNNParameters():
    def __init__(self, params):
        self.bidirectional = False
        
        if "forward_layers" not in params:
            raise Exception("A forward layer must be specified.")

        if not is_list(params["forward_layers"]):
            raise Exception("Forward RNN layers must be a list")
        num_fw_layers = len(params["forward_layers"])
        self.forward_layers = list()                                                
        for i in range(num_fw_layers):
            self.forward_layers.append(LayerParameters(params["forward_layers"][i]))

        if "backward_layers" in params:
            if not is_list(params["backward_layers"]):
                raise Exception("Backward RNN layers must be a list")
            self.bidirectional = True
            num_bw_layers = len(params["backward_layers"])
            self.backward_layers = list()                                                
            for i in range(num_bw_layers):
                self.backward_layers.append(LayerParameters(params["backward_layers"][i]))

class LayerParameters():
    def __init__(self, params):
        self.cell_class = params["cell_class"]
        self.num_units = params["num_units"]
        
        if "dropout_input_keep_prob" in params:
            if isinstance(params["dropout_input_keep_prob"], float):
                self.dropout_input_keep_prob = params["dropout_input_keep_prob"]
            else:
                raise Exception("dropout_input_keep_prob must be float")
        else:
            self.dropout_input_keep_prob = 1.0

        if "dropout_output_keep_prob" in params:
            if isinstance(params["dropout_output_keep_prob"], float):
                self.dropout_output_keep_prob = params["dropout_output_keep_prob"]
            else:
                raise Exception("dropout_output_keep_prob must be float")
        else:
            self.dropout_output_keep_prob = 1.0

        if "dropout_state_keep_prob" in params:
            if isinstance(params["dropout_state_keep_prob"], float):
                self.dropout_state_keep_prob = params["dropout_state_keep_prob"]
            else:
                raise Exception("dropout_state_keep_prob must be float")
        else:
            self.dropout_state_keep_prob = 1.0


class OptimizerParameters():
    def __init__(self, params):
        self.name = params["name"]
        self.learning_rate = params["learning_rate"]
        
        if not (("decay_factor" in params) == ("decay_step" in params)):
            Exception("Both or none of decay_factor and decay_step must be specified")
            
        if "decay_factor" in params:
            self.decay_factor = params["decay_factor"]
            self.decay_step = params["decay_step"]
        else:
            self.decay_factor = 1.0
            self.decay_step = 10000


class DataSources():
    def __init__(self, params, model):
        if len(params["inputs"]) != len(model.inputs):
            raise Exception("Number of inputs does not match")
        if len(params["outputs"]) != len(model.outputs):
            raise Exception("Number of outputs does not match")

        if not is_list(params["inputs"]) or not is_list(params["outputs"]):
            raise Exception("Inputs and outputs must be lists")

        self.inputs = list()
        for i in range(len(model.inputs)):
            self.inputs.append(params["inputs"][i])

        self.outputs = list()
        for i in range(len(model.outputs)):
            self.outputs.append(params["outputs"][i])

