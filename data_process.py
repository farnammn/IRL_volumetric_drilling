from abc import ABC, abstractmethod
class Dataset(ABC):
    @abstractmethod
    def return_data_list(self):
        pass

class GymSet(Dataset):
    def return_data_list(self):
        pass


class SimDrivingSet(Dataset):
    def return_data_list(self):
        pass