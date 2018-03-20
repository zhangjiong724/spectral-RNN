import json
import math

"""
Parameter class
"""
class Params (object):
    def __init__(self):
        self.cell = None # RNN cell
        self.initial_learning_rate = math.exp(-10) # learning rate for SGD, [exp(-10), 1]
        self.lr_decay = 0.8 # the multiplier to multiply the learning rate by every 1k iterations, in range of [0.8, 0.999]
        self.num_epochs = 100 # number of epochs
        self.dropout_keep_rate = 0.5 # percent of output units that are kept during dropout, in range (0, 1]
        self.num_units = 200 # number of units
        self.num_layers = 1 # number of units
        self.r_size = 60 # the number of reflectors used in svdRNN
        self.r_margin = 0.01 # the number of reflectors used in svdRNN
        self.time_steps = None # time steps, time_steps*input_size = sequence length
        self.input_size = None # dimensionality of input features at each time step
        self.output_size = None # dimensionality of label
        self.gpu_flag = True # use GPU or not
        self.random_seed = 1000 # random seed
        self.dataset = 'mnist.28' # dataset name
        self.batch_size = 128 # batch size
        self.regression_flag = True # regression or classification
        self.model_dir = '' # directory to save model, will append .cell_name
        self.load_model_dir = '' # directory to save model, will append .cell_name
        self.pred_dir = '' # directory for prediction results, will append .dataset.cell_name.[Xy]
        self.load_model = False # load model or not
        self.train_flag = True # train model or not
        self.batch_norm = False # batch normalization or not
        self.display_epoch_num = 1 # display how many evaluations per epoch
    """
    convert to json
    """
    def toJson(self):
        data = dict()
        data['cell'] = self.cell
        data['initial_learning_rate'] = self.initial_learning_rate
        data['lr_decay'] = self.lr_decay
        data['num_epochs'] = self.num_epochs
        data['dropout_keep_rate'] = self.dropout_keep_rate
        data['num_units'] = self.num_units
        data['num_layers'] = self.num_layers
        data['r_size'] = self.r_size
        data['r_margin'] = self.r_margin
        data['time_steps'] = self.time_steps
        data['input_size'] = self.input_size
        data['output_size'] = self.output_size
        data['gpu_flag'] = self.gpu_flag
        data['batch_size'] = self.batch_size
        data['random_seed'] = self.random_seed
        data['dataset'] = self.dataset
        data['regression_flag'] = self.regression_flag
        data['model_dir'] = self.model_dir
        data['load_model_dir'] = self.load_model_dir
        data['pred_dir'] = self.pred_dir
        data['load_model'] = self.load_model
        data['train_flag'] = self.train_flag
        data['batch_norm'] = self.batch_norm
        data['display_epoch_num'] = self.display_epoch_num
        return data
    """
    load form json
    """
    def fromJson(self, data):
        if 'cell' in data: self.cell = data['cell']
        if 'initial_learning_rate' in data: self.initial_learning_rate = data['initial_learning_rate']
        if 'lr_decay' in data: self.lr_decay = data['lr_decay']
        if 'num_epochs' in data: self.num_epochs = data['num_epochs']
        if 'dropout_keep_rate' in data: self.dropout_keep_rate = data['dropout_keep_rate']
        if 'num_units' in data: self.num_units = data['num_units']
        if 'num_layers' in data: self.num_layers = data['num_layers']
        if 'r_size' in data: self.r_size = data['r_size']
        if 'r_margin' in data: self.r_margin = data['r_margin']
        if 'time_steps' in data: self.time_steps = data['time_steps']
        if 'input_size' in data: self.input_size = data['input_size']
        if 'output_size' in data: self.output_size = data['output_size']
        if 'gpu_flag' in data: self.gpu_flag = data['gpu_flag']
        if 'batch_size' in data: self.batch_size = data['batch_size']
        if 'random_seed' in data: self.random_seed = data['random_seed']
        if 'dataset' in data: self.dataset = data['dataset']
        if 'regression_flag' in data: self.regression_flag = data['regression_flag']
        if 'model_dir' in data: self.model_dir = data['model_dir']
        if 'load_model_dir' in data: self.load_model_dir = data['load_model_dir']
        if 'pred_dir' in data: self.pred_dir = data['pred_dir']
        if 'load_model' in data: self.load_model = data['load_model']
        if 'train_flag' in data: self.train_flag = data['train_flag']
        if 'batch_norm' in data: self.batch_norm = data['batch_norm']
        if 'display_epoch_num' in data: self.display_epoch_num = data['display_epoch_num']

    """
    dump to json file
    """
    def dump(self, filename):
        with open(filename, 'w') as f:
            meta = self.toJson()
            json.dump(dict((key, value) for key, value in meta.iteritems() if value != None), f)
    """
    load from json file
    """
    def load(self, filename):
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))
    """
    string
    """
    def __str__(self):
        return str(self.toJson())
    """
    print
    """
    def __repr__(self):
        return self.__str__()
