import os
import json
import argparse

import torch

from core.tester import Tester

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs/youtube-vos.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument('-e', '--exam', action='store_true')
parser.add_argument("--ds_name", type=str, required=True)
args = parser.parse_args()

def main_worker(config):
    config["device"] = torch.device("cuda:{}".format(config["device_int"]))
    print("Device: ", config["device"])

    config["save_dir"] = os.path.join(config["save_dir"], '{}_{}'.format(config["model"], os.path.basename(args.config).split('.')[0]))

    os.makedirs(config['results_dir'], exist_ok=True)

    print(config)

    tester = Tester(config)
    tester.test()


if __name__ == "__main__":
    config = json.load(open(args.config))
    config['model'] = args.model
    config['config'] = args.config

    config["ckpt"] = args.ckpt
    config["ds_name"] = args.ds_name

    config["world_size"]=1    

    main_worker(config)