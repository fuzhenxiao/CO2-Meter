import torch
import numpy as np
from torch_geometric.data import Dataset


# torch.save(data_list, output_root + "latency_data.pt")
# torch.save(data_list2, output_root + "energy_data.pt")
# #torch.save(global_feature, output_root + "global_feature.pt")


class CarbonDataset(Dataset):
    def __init__(
        self,
        root,
        input_file,
        global_file,
        length_file,
        transform=None,
    ):
        super(CarbonDataset, self).__init__(root=root, transform=transform)
        self.root = root
        self.input_file = input_file
        self.global_file = global_file
        self.length_file = length_file
        self.big_data = torch.load(self.root + self.input_file, weights_only=False) # may need to add 'weights_only = False' if error occurs
        self.global_data = torch.load(self.root + self.global_file , weights_only=False)
        with open( self.root + self.length_file, mode="r") as ref:
            for line in ref:
                self.total_length = int(line.rstrip())

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return self.total_length

    def get(self, idx):
        return self.big_data[idx], self.global_data[idx]
