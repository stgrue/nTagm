import numpy as np

from yaml_parser import Input, Output

def compute_dimensionality(src):
    if isinstance(src, Input):
        if src.type == "vocabulary":
            j = 0
            with open(src.vocab_file_path, encoding="utf-8") as f:
                for line in f:
                    j += 1
            j += len(src.additional_labels)
            return j
        elif src.type == "embeddings":
            with open(src.embeds_file_path, encoding="utf-8") as f:
                line = next(f)
                line = line.strip().split(' ')
                return len(line) - 1
    elif isinstance(src, Output):
        j = 0
        with open(src.vocab_file_path, encoding="utf-8") as f:
            for line in f:
                j += 1
        return j        
    else:
        raise Exception("Can only compute dimensionality of inputs and outputs")

    
def create_lookup_matrix(_input):
    if _input.type == "vocabulary":
        num_lines = 0
        with open(_input.vocab_file_path, encoding="utf-8") as f:
            for line in f:
                num_lines += 1
        return np.identity(num_lines + len(_input.additional_labels)) # One-hot encoding              
    elif _input.type == "embeddings":
        embeds = list()
        with open(_input.embeds_file_path, encoding="utf-8") as f:
            for line in f:
                line = line.split(' ')
                vals = list(map(float, line[1:]))
                embeds.append(vals)
            for label in _input.additional_labels:
                vals = np.random.normal(scale=0.5, size=(len(embeds[0]),)) 
                embeds.append(vals)
        return np.array(embeds) # Embedding matrix
    else:
        raise Exception("Unknown input type '{}'".format(_input.type))


def load_mappings(model):
    input_labels_to_ix = list()
    input_ix_to_labels = list()
    output_labels_to_ix = list()
    output_ix_to_labels = list()
    
    for i in range(len(model.inputs)):
        input_labels_to_ix.append(dict())
        input_ix_to_labels.append(dict())
        j = 0
        if model.inputs[i].type == "vocabulary":
            with open(model.inputs[i].vocab_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    input_labels_to_ix[i][line] = j
                    input_ix_to_labels[i][j] = line
                    j += 1     
        elif model.inputs[i].type == "embeddings":
            with open(model.inputs[i].embeds_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.split(' ')
                    label = line[0]
                    input_labels_to_ix[i][label] = j
                    input_ix_to_labels[i][j] = label
                    j += 1              
        else:
            assert False # Handle other input types here
        for label in model.inputs[i].additional_labels:
            input_labels_to_ix[i][label] = j
            input_ix_to_labels[i][j] = label
            j += 1

    for i in range(len(model.outputs)):
        output_labels_to_ix.append(dict())
        output_ix_to_labels.append(dict())
        j = 0
        with open(model.outputs[i].vocab_file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                output_labels_to_ix[i][line] = j
                output_ix_to_labels[i][j] = line
                j += 1
            
    return input_labels_to_ix, input_ix_to_labels, output_labels_to_ix, output_ix_to_labels


class SequenceBatchContainer():
    def __init__(self, input_batches, output_batches):
        self.input_batches = input_batches
        self.output_batches = output_batches
        for inp_batch in self.input_batches:
            assert len(inp_batch[0]) == len(self.input_batches[0][0])
        for outp_batch in self.output_batches:
            assert len(outp_batch[0]) == len(self.input_batches[0][0])
        
def load_sequences(data_source, mappings):
    '''Load batches of sequences from the specified source.
       Labels are converted to numbers according to given mappings.'''
    input_labels_to_ix, _, output_labels_to_ix, _ = mappings
    sequences = list()

    input_files = list()
    output_files = list()
    for inp in data_source.inputs:
        input_files.append(open(inp, encoding="utf-8"))
    for outp in data_source.outputs:
        output_files.append(open(outp, encoding="utf-8"))

    for global_line in input_files[0]: # Iterate over all files simultaneously
        input_lines = list()
        output_lines = list()

        global_line = global_line.strip().split('\t')
        global_line = labels_as_numbers(global_line, input_labels_to_ix[0])
        input_lines.append([global_line]) # TODO: Batching/padding (batch size hardcoded to 1 for now)
        
        for i in range(1, len(input_files)):
            line = next(input_files[i]) # TODO when batching: Pad several sequences to the same length and check via assertion
            line = line.strip().split('\t')
            line = labels_as_numbers(line, input_labels_to_ix[i])
            input_lines.append([line])
        for i in range(len(output_files)):
            line = next(output_files[i])
            line = line.strip().split('\t')
            line = labels_as_numbers(line, output_labels_to_ix[i])
            output_lines.append([line])
        sequences.append(SequenceBatchContainer(input_lines, output_lines))
        
    for i_f in input_files:
        i_f.close()
    for o_f in output_files:
        o_f.close()

    return sequences


def labels_as_numbers(seq, lbl_to_ix):
    return [lbl_to_ix[lbl] for lbl in seq]

def numbers_as_labels(seq, ix_to_lbl):
    return [ix_to_lbl[ix] for ix in seq]
