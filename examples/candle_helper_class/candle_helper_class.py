from __future__ import print_function

import argparse
import configparser
import os

from keras import backend as K

# shared modules
import p1_common

file_path = os.path.dirname(os.path.realpath(__file__))

class CandleHelperClass(object):
    def __init__(self):
        self.default_config_file = ''

    def read_config_file(self, file):
        config = configparser.ConfigParser()
        config.read(file)
        section = config.sections()
        fileParams = {}

        fileParams['epochs'] = eval(config.get(section[0],'epochs'))
        fileParams['batch_size'] = eval(config.get(section[0],'batch_size'))
        fileParams['learning_rate'] = eval(config.get(section[0], 'learning_rate'))
        fileParams['save'] = eval(config.get(section[0], 'save'))

        # parse the remaining values
        for k,v in config.items(section[0]):
            if not k in fileParams:
                fileParams[k] = eval(v)

        return fileParams

    def initialize_parameters(self):
        # can we  merge get_local_parser(), read_config_file(), and common_parser() chain ?
        # how to config with _set.R for HPO?
        parser = argparse.ArgumentParser(description='CANDLE class parser')

        parser.add_argument("--config_file", dest='config_file', type=str,
                            default=os.path.join(file_path, self.default_config_file),
                            help="specify model configuration file")

        parser = p1_common.get_default_neon_parse(parser)
        parser = p1_common.get_p1_common_parser(parser)

        args = parser.parse_args()
        fileParameters = self.read_config_file(args.config_file)
        gParameters = p1_common.args_overwrite_config(args, fileParameters)
        return gParameters

    def run(self, gParameters):
        print('please implement run method')

    def main(self):
        gParameters = self.initialize_parameters()
        self.run(gParameters)

        try:
            K.clear_session()
        except AttributeError:      # theano does not have this function
            pass

if __name__ == '__main__':
    c = CandleHelperClass()
    c.main()
