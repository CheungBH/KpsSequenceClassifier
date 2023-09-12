import torch
import sys
sys.path.append("../../")
from src.TCN.TCNsrc.model import TCN
import numpy as np
import os
import config

device = config.device
params_ls = config.TCN_structure


class prediction:
    def __init__(self, model_name):
        structre_num = int(model_name.split('_')[1][6:])
        self.model = TCN(input_size=34, output_size=2, num_channels= params_ls[structre_num][0], kernel_size=params_ls[structre_num][1], dropout=0).cuda()
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)#(1, 30, 34)
        data = data.permute(0,2,1)
        return data#(1, 34, 30)

    def predict_pre_second(self, data):
        input = self.get_input_data(data)
        output = self.model(input)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred

    # def predict(self):
    #     preds = []
    #     for i in range(0, self.input_data.shape[0], 30):
    #         data = self.get_input_data(self.input_data[i:i+30,:])
    #         if data.size(1)<30:
    #             break
    #         pred = self.predict_pre_second(data)
    #         print('pred:',pred)
    #         preds.append(pred)
    #     return preds


if __name__ == '__main__':
    input_pth = 'data.txt'
    model_name = 'TCN_struct1_10-10-10-10-10-10.pth'
    inp = np.loadtxt(input_pth).astype(np.float32).reshape(-1, 34)
    prediction = prediction(model_name)
    res = prediction.predict_pre_second(inp)
    print(res)
