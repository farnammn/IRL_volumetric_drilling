from preprocess import DataSet

file_list = [
    "20220407_170930.hdf5",
    "20220407_171452.hdf5",
    "20220407_171709.hdf5",
    "20220407_171806.hdf5",
    "20220407_171928.hdf5"
]

data_set = DataSet(file_list, 100)
data_set.next_batch()

