


import sys
import torch
from Backbone import DAL_model
from train import Trainer
from meta import cacd, age_cutoffs
from utils import path2age
# pytohn3 main.py ctx_id epochs path2wgts      # ctx_id: -1 for cpu, 0 for gpu 0

def main():
    dataset = cacd
    model = DAL_model('cosface', dataset['n_cls'])
    model.load_state_dict(torch.load("../cache/dal-pretrained-1st-stage.pth"))
    # return
    if len(sys.argv) >= 4:
        model.load_state_dict(torch.load(sys.argv[3]))
        print(f'Loaded weights: {sys.argv[3]}')
        start_epoch = path2age(sys.argv[3], '_|\.', 0) + 1
    else:
        start_epoch = 0
    trainer = Trainer(model, dataset, int(sys.argv[1]), print_freq=100, train_head_only=False)
    save = '../cache/model_cache/'
    trainer.train(int(sys.argv[2]), start_epoch, save)

if __name__ == '__main__':
    main()