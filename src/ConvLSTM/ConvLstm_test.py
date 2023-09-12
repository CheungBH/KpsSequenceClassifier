import torch
import sys
sys.path.append("../../")
from src.ConvLSTM.model import ConvLSTM
import numpy as np
import os
from config import config

device = config.device
params_ls = config.ConvLSTM_structure
kps_num = config.test_kps_num


class prediction(object):
    def __init__(self, model_name, num_class=2):
        # self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,1,17,2)#(seq_l, 1, 17, 2)
        structure_num = int((model_name.split("/")[-1]).split('_')[1][6:])
        self.model = ConvLSTM(input_size=(int(kps_num/2), 2),
                             input_dim=1,
                             hidden_dim=params_ls[structure_num][0],
                             kernel_size=params_ls[structure_num][1],
                             num_layers=len(params_ls[structure_num][0]),
                             num_classes=num_class,
                             batch_size=2,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False,
                             attention=params_ls[structure_num][2]).cuda()
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        return data#(1, 30, 1, 17, 2)

    def predict_pre_second(self, data):
        input = self.get_input_data(data.reshape(-1,1,17,2))
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred

    # def predict(self):
    #     preds = []
    #     for i in range(0, self.input_data.shape[0], 30):
    #         data = self.get_input_data(self.input_data[i:i+30,:,:,:])
    #         if data.size(1)<30:
    #             break
    #         pred = self.predict_pre_second(data)
    #         print('pred:',pred)
    #         preds.append(pred)
    #     return pred


if __name__ == '__main__':
    model = 'ConvLSTM_struct2_10-10-10-10-10-99.pth'
    input_pth = 'data.txt'
    inp = np.loadtxt(input_pth).astype(np.float32)
    pred = prediction(model)
    res = pred.predict_pre_second(inp)
    print(res)
    