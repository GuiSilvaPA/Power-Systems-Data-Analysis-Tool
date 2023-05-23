from NTW_Reader import NTW_Reader
import numpy as np



class NTW_Editor(NTW_Reader):

    def change_load(self, param, multi, spec=None, arre=2, keepFP=False):
        
        
        ite = spec if spec else [i for i in range(len(self.load_data))]
        
        if keepFP:
            for i in ite:      
                
                kw, kvar = self.load_data['PL_MW'].values[i], self.load_data['QL_MVAR'].values[i]
                FP       = np.cos(np.arctan(kvar/kw))
                
                kw_new   = kw*multi
                kvar_new = np.tan(np.arccos(FP))*kw_new
                
                self.load_data.loc[i, 'PL_MW']   = kw_new
                self.load_data.loc[i, 'QL_MVAR'] = kvar_new             

        else:      
            for i in ite:
                self.load_data[param][i] = round(float(self.load_data.iloc[i][param])*multi, arre)    
                
        self.cargaTotal = 0
        for i in self.load_data['PL_MW'].values:
            self.cargaTotal += float(i)

    def save(self, save_path):
        
        # GEN
        
        for idx, i in enumerate(range(self.f_gen, self.l_gen+1)):
            
            new_line = ''
            for j in range(len(self.gen_data.iloc[idx])):
                if j != len(self.gen_data.iloc[idx])-1:
                    
                    if len(str(self.gen_data.iloc[idx, j])) < 5:
                        new_line += f'{str(self.gen_data.iloc[idx, j]): >6}' + ','
                    elif len(str(self.gen_data.iloc[idx, j])) >= 5 and len(str(self.gen_data.iloc[idx, j])) < 10:
                        new_line += f'{str(self.gen_data.iloc[idx, j]): >11}' + ','
                    else:
                        new_line += f'{str(self.gen_data.iloc[idx, j]): >14}' + ','
                    
                    
                    
                    #new_line += str(self.gen_data.iloc[idx, j]) + ','
                else:
                    new_line += '  ' + str(self.gen_data.iloc[idx, j]) + '/'
                
            self.lines[i] = new_line + ' \n'
            
        # LOAD
            
        for idx, i in enumerate(range(self.f_load, self.l_load+1)):
            
            new_line = ''
            for j in range(len(self.load_data.iloc[idx])):
                if j != len(self.load_data.iloc[idx])-1:
                    
                    if len(str(self.load_data.iloc[idx, j])) < 5:
                        new_line += f'{str(self.load_data.iloc[idx, j]): >6}' + ','
                    elif len(str(self.load_data.iloc[idx, j])) >= 5 and len(str(self.load_data.iloc[idx, j])) < 10:
                        new_line += f'{str(self.load_data.iloc[idx, j]): >11}' + ','
                    else:
                        new_line += f'{str(self.load_data.iloc[idx, j]): >14}' + ','
                    
                else:
                    new_line += '  ' + str(self.load_data.iloc[idx, j]) + '/'
                
            self.lines[i] = new_line + ' \n'
            
        # SAVE
            
        with open(save_path, 'w') as f:
            for line in self.lines:
                f.write(line)






if __name__ == '__main__':

    path = 'C:/Users/albing-local/Desktop/Data/DataForTest/9bus.ntw'

    NE = NTW_Editor(path)
    NE.change_gen()