import numpy as np
import pandas as pd


class DatasetClass:
    def __init__(self, data_path, norm_data:bool=False):
        self.path = data_path
        if norm_data:
            self.x_label = "nx"
            self.y_label = "ny"
        else:
            self.x_label = "x"
            self.y_label = "y"
    
    def __reflect_index(self, data, index):
        if index != None:
            data = data[:, index]
        return data
        
    def __load_df(self, data_label):
        data_x = pd.read_csv(f"{self.path}/{data_label}_{self.x_label}.csv", index_col=0)
        data_y = pd.read_csv(f"{self.path}/{data_label}_{self.y_label}.csv", index_col=0)
        return data_x, data_y
    
    def __load_data(self, data_label, x_index, y_index):
        data_x, data_y = self.__load_df(data_label)
        data_x = self.__reflect_index(data_x.values, x_index)
        data_y = self.__reflect_index(data_y.values, y_index)
        return data_x, data_y
    
    def __load_stack(self, dataset_list, x_index, y_index):
        for label in dataset_list:
            tmp_x, tmp_y = self.__load_data(label, x_index, y_index)
            if dataset_list.index(label) == 0:
                data_x = tmp_x
                data_y = tmp_y
            else:
                data_x = np.vstack((data_x, tmp_x))
                data_y = np.vstack((data_y, tmp_y))
        return data_x, data_y
    
    def __load_dict(self, dataset_list, x_index, y_index):
        data_x, data_y = {}, {}
        for label in dataset_list:
            tmp_x, tmp_y = self.__load_data(label, x_index, y_index)
            data_x[label] = tmp_x
            data_y[label] = tmp_y
        return data_x, data_y
    
    def get_data(self, dataset_label, x_index=None, y_index=None, dict_type:bool=False):
        if not dict_type:
            if type(dataset_label) == str:
                data_x, data_y = self.__load_data(dataset_label, x_index, y_index)
            else:
                data_x, data_y = self.__load_stack(dataset_label, x_index, y_index)
        else:
            data_x, data_y = self.__load_dict(dataset_label, x_index, y_index)
        return data_x, data_y
    
    def get_dataframe(self, dataset_list):
        data_x = {}
        data_y = {}
        for label in dataset_list:
            tmp_x, tmp_y = self.__load_df(label)
            data_x[label] = tmp_x
            data_y[label] = tmp_y
        return data_x, data_y
    