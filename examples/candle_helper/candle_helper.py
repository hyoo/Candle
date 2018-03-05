from __future__ import print_function

import argparse
import configparser
import os

from keras import backend as K

# shared modules
import p1_common

file_path = os.path.dirname(os.path.realpath(__file__))
default_config_file = ''

def common_parser(parser):
    parser.add_argument("--config_file", dest='config_file', type=str,
                        default=os.path.join(file_path, default_config_file),
                        help="specify model configuration file")

    parser = p1_common.get_default_neon_parse(parser)
    parser = p1_common.get_p1_common_parser(parser)

    return parser

def get_local_parser():
    # print('please implement get_local_parser method')
    parser = argparse.ArgumentParser(description='CANDLE framework parser')
    return common_parser(parser)

def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    section = config.sections()
    fileParams = {}

    # fileParams['data_url'] = eval(config.get(section[0],'data_url'))
    # fileParams['train_data'] = eval(config.get(section[0],'train_data'))
    # fileParams['test_data'] = eval(config.get(section[0],'test_data'))
    # fileParams['model_name'] = eval(config.get(section[0],'model_name'))
    # fileParams['conv'] = eval(config.get(section[0],'conv'))
    # fileParams['dense'] = eval(config.get(section[0],'dense'))
    # fileParams['activation'] = eval(config.get(section[0],'activation'))
    # fileParams['out_act'] = eval(config.get(section[0],'out_act'))
    # fileParams['loss'] = eval(config.get(section[0],'loss'))
    # fileParams['optimizer'] = eval(config.get(section[0],'optimizer'))
    # fileParams['metrics'] = eval(config.get(section[0],'metrics'))
    fileParams['epochs'] = eval(config.get(section[0],'epochs'))
    fileParams['batch_size'] = eval(config.get(section[0],'batch_size'))
    fileParams['learning_rate'] = eval(config.get(section[0], 'learning_rate'))
    # fileParams['drop'] = eval(config.get(section[0],'drop'))
    # fileParams['classes'] = eval(config.get(section[0],'classes'))
    # fileParams['pool'] = eval(config.get(section[0],'pool'))
    fileParams['save'] = eval(config.get(section[0], 'save'))

    # parse the remaining values
    for k,v in config.items(section[0]):
        if not k in fileParams:
            fileParams[k] = eval(v)

    return fileParams

def initialize_parameters():
    # print('please implement initialize_parameters method')
    parser = get_local_parser()
    args = parser.parse_args()
    fileParameters = read_config_file(args.config_file)
    gParameters = p1_common.args_overwrite_config(args, fileParameters)
    return gParameters

def run(gParameters):
    print('please implement run method')
    print ('Params:', gParameters)

def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
