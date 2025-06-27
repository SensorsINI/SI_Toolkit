from SI_Toolkit.Functions.General.Dataset import DatasetTemplate
from torch.utils import data


class Dataset(DatasetTemplate, data.Dataset):
    def __init__(self,
                 dfs,
                 args,
                 inputs=None,
                 outputs=None,
                 exp_len=None,
                 shuffle=True,
                 normalization_info=None,
                 ):
        super(Dataset, self).__init__(dfs, args, inputs, outputs, exp_len, shuffle,
                                      normalization_info=normalization_info)

    def __len__(self):
        return self.number_of_samples_to_use

    def __getitem__(self, idx):
        return self.get_series(idx)
