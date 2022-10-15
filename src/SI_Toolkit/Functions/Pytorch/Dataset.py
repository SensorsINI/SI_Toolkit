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
                 ):
        super(Dataset, self).__init__(dfs, args, inputs, outputs, exp_len, shuffle)

        if self.tiv:
            # FIXME: tiv not working here for Pytorch as it requires change at each epoch end
            raise Exception('TIV in Pytorch might now work as expected.'
                            'If you know what you do comment this line out (but do not commit!)'
                            'Otherwise provide as TIV empty list.')

    def __len__(self):
        return self.number_of_samples_to_use

    def __getitem__(self, idx):
        return self.get_series(idx, get_time_axis=False)
