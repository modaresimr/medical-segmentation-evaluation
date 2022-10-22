from .dataset import Dataset
from .nib import read_nib


class NibDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

    def get_CT(self, id):
        return read_nib(get_file(self.dataset_info[CT][f'{id}']))

    def get_groundtruth(self, id):
        return read_nib(get_file(self.dataset_info[GT][f'{id}']))

    def get_prediction(self, method, id):
        return read_nib(get_file(self.dataset_info[PREDS][method][f'{id}']))
