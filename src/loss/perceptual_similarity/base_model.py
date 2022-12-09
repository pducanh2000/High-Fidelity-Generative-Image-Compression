import os
import torch
import numpy as np


class BaseModel(object):
    def __init__(self):
        self.name = "Base_model"
        self.use_gpu = None
        self.gpu_ids = None

        self.image_paths = None
        self.save_dir = None
        pass

    def name(self):
        return self.name

    def initialize(self, use_gpu=True, gpu_ids=(0, )):
        self.use_gpu = use_gpu
        self.gpu_ids = list(gpu_ids)

    def forward(self):
        pass

    def get_images_path(self):
        pass

    def optimizer_parameter(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_errors(self):
        pass

    def save(self, label):
        pass

    def save_network(self, network, save_folder, network_label, epoch_label):
        save_file_name = f'{epoch_label}_net_{network_label}'
        save_path = os.path.join(save_folder, save_file_name)
        torch.save(network.state_dict(), save_path)

    def load_network(self, network, network_label, epoch_label):
        save_filename = f'{epoch_label}_net_{network_label}'
        save_path = os.path.join(self.save_dir, save_filename)
        print(f'Loading network from {save_path}')
        network.load_state_dict(torch.load(save_path))

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'), flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'), [flag, ], fmt='%i')
