import pandas as pd

import json

from tqdm import tqdm


class DynamicData():
    
    def __init__(self, path, other=False):

        with open(path) as f:
            self.lines = f.readlines()

        print('Teste')

        # lines = [line.strip().replace('/', '').replace('!', '') for line in lines]
        # lines = [line for line in lines if line != '']
        # lines = lines[1:-3]

        # SM_index = [idx for idx, line in enumerate(lines) if 'SM' in line]
        # SM_index.append(len(lines))

        # grouped = [lines[SM_index[i]+1 : SM_index[i+1]] for i in range(len(SM_index)-1)]

        # print(grouped[0][0])

        # ident = [gen[1].split() for gen in grouped]
        # gen   = [gen[3].split() for gen in grouped]
        # # avr   = self.excitation(grouped, 5, 'AVR')
        # # pss   = self.excitation(grouped, 7, 'PSS')
        # # gov   = self.excitation(grouped, 9, 'GOV')

        # ident_col = ['Bus', 'ID', 'AVR', 'PSS', 'UEL', 'OEL', 'SCL', 'Gov', 'Ctrl', 'Rc(pu)', 'Xc(pu)', 'Tr(s)', 'Bus-Name', 'CtrBus-Name']
        # gen_col   = ['Xd(pu)', 'Xld(pu)', 'Xlld(pu)', 'Xq(pu)', 'Xlq(pu)', 'Xllq(pu)', 'Ra(pu)', 'Base(MVA)', 'Xl(pu)', 'Xt(pu)', 'Tld(s)', 'Tlld(s)', 'Tlq(s)', 'H(MWMVA.s)', 'D(pupu)', 'Tllq(pu)', 'S1.0', 'S1.2', 'Cg']
        # # avr_col   = ['KP', 'KI', 'KD', 'TD', 'VRMAX', 'VRMIN', 'KA', 'TA', 'KE', 'TE', 'KF', 'TF', 'VEMIN', 'AEX', 'BEX', 'PIDMAX', 'PIDMIN', 'OELFLAG', 'UELFLAG']
        # # pss_col   = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'T1', 'T2', 'T3', 'T4', 'T5', 'K', 'Vmin', 'Vmax', 'Vcu', 'Vcl', 'Type']
        # # gov_col   = ['R', 'T1', 'PMAX', 'PMIN', 'T2', 'T3']

        # self.ident_data = pd.DataFrame(data=ident, columns=ident_col)
        # self.gen_data   = pd.DataFrame(data=gen, columns=gen_col)

    def _find_models(self, save=False):

        self.models = {}
        for idx, line in enumerate(self.lines):

            line = line.split()

            if line == []:
                continue

            if line[0].startswith('!') or line[0].startswith(',') or line[0].startswith('VERSION')  or '/' in line[0]:
                continue

            try:
                float(line[0])

            except:

                self.models[idx+1] = line
        
        if save:
            with open("models.json", "w") as f:
                json.dump(self.models, f, indent=4)

    
    def _find_SM(self, models_path=None):

        if models_path is None:
            self._find_models()

        else:            
            with open(models_path, "r") as fp:
                self.models = json.load(fp)

        SM_models = {}
        for key, value in tqdm(zip(self.models.keys(), self.models.values())):

            if 'SM' in value[0]:
                infos = []
                for idx in range(key-1, len(self.lines)):

                    line = self.lines[idx].split()

                    if line == []:
                        continue

                    if line[0].startswith('!'):
                        continue

                    try:
                        float(line[0])

                    except:
                        continue

                    infos.append(self.lines[idx])

                SM_models[key] = infos

        with open("SM_models.json", "w") as f:
            json.dump(SM_models, f, indent=4)




        



if __name__ == '__main__':

    path = 'DataForTest/SIN.dyn'

    DD = DynamicData(path)
    DD._find_SM()

    # DD.ident_data.to_excel("ident_data.xlsx") 
    # ND.load_data.to_excel("load_data.xlsx")
    # ND.gen_data.to_excel("gen_data.xlsx")

    # ND.concat()
    # ND.data.to_excel("data.xlsx")

