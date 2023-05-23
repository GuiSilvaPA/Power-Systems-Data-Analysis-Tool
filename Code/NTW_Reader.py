import pandas as pd

class NTW_Reader():
    
    def __init__(self, path, other=False):
        
        self.path  = path
        self.time  = 15.0
        self.other = other
        
        with open(path) as f:
            self.lines = f.readlines()

        self.f_bus = 4

        for idx, line in enumerate(self.lines):

            if 'END OF BUS DATA'  in line: self.l_bus  = idx - 1

            if 'BEGIN LOAD DATA'  in line: self.f_load = idx + 2
            if 'END OF LOAD DATA' in line: self.l_load = idx - 1

            if 'BEGIN GENERATOR DATA'  in line: self.f_gen = idx + 2
            if 'END OF GENERATOR DATA' in line: self.l_gen = idx - 1

            if 'BEGIN TRANSMISSION LINE DATA'  in line: self.f_tra = idx + 2
            if 'END OF TRANSMISSION LINE DATA' in line: self.l_tra = idx - 1

            if 'BEGIN TRANSFORMER DATA'  in line: self.f_trs = idx + 1
            if 'END OF TRANSFORMER DATA' in line: self.l_trs = idx - 1

        self._get_bus_data()
        self._get_load_data()
        self._get_gen_data()
        self._get_transmission_data()
        self._get_transformer_data()

    # Get the BUS's DATA ======================================================================================================================================

    def _get_bus_data(self):      

        columns = self.lines[self.f_bus-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split() #
        data = []

        for i in range(self.f_bus, self.l_bus + 1):
            bus_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(' ', '').split(',') #.replace(',', ' ')
            data.append(bus_info)

        try:
            self.bus_data = pd.DataFrame(data, columns=columns)
            
            for col in self.bus_data.columns:
                try:
                    self.bus_data[col] = self.bus_data[col].astype('float')

                except:
                    pass

        except:
            self.bus_data = pd.DataFrame(data)
            print('BUS: Check the data or the columns')
            print(columns)


    # Get the LOAD's DATA =====================================================================================================================================

    def _get_load_data(self):      

        columns = self.lines[self.f_load-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_load, self.l_load + 1):
            load_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(' ', '').split(',')
            data.append(load_info)

        try:
            self.load_data = pd.DataFrame(data, columns=columns)
            
            for col in self.load_data.columns:
                try:
                    self.load_data[col] = self.load_data[col].astype('float')

                except:
                    pass

        except:
            self.load_data = pd.DataFrame(data)
            print('LOAD: Check the data or the columns')

    # Get the GENERATION's DATA ===============================================================================================================================

    def _get_gen_data(self):      

        columns = self.lines[self.f_gen-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_gen, self.l_gen + 1):
            gen_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(' ', '').split(',')
            data.append(gen_info)

        try:
            self.gen_data = pd.DataFrame(data, columns=columns)

            for col in self.gen_data.columns:
                try:
                    self.gen_data[col] = self.gen_data[col].astype('float')

                except:
                    pass

        except:
            self.gen_data = pd.DataFrame(data)
            print('GEN: Check the data or the columns')

    # Get the TRANSMISSION's DATA ===============================================================================================================================

    def _get_transmission_data(self):      

        columns = self.lines[self.f_tra-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_tra, self.l_tra + 1):
            tra_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(' ', '').split(',')
            data.append(tra_info)

        try:
            self.transmission_data = pd.DataFrame(data, columns=columns)

        except:
            try:

                hard_col = ['BFR_ID', 'BTO_ID', 'CI', 'RL_PU', 'XL_PU', 'B_MVAR', 'L1_MVA', 'L2_MVA', 'L3_MVA',
                            'BFR_LB', 'BTO_LB', 'LEN_KM', 'ARE', 'OWN', 'ZE_R', 'ZE_X', 'ZE_C', 'BR_NAME',
                            'BUS_CTR1', 'CTR1', 'BUS_CTR2', 'CTR2',
                            'BFR_SH1', 'GSHT_F1', 'BSHT_F1','BTO_SH1', 'GSHT_T1', 'BSHT_T1',
                            'BFR_SH2', 'GSHT_F2', 'BSHT_F2','BTO_SH2', 'GSHT_T2', 'BSHT_T2',
                            'BFR_SH3', 'GSHT_F3', 'BSHT_F3','BTO_SH3', 'GSHT_T3', 'BSHT_T3']
                self.transmission_data = pd.DataFrame(data, columns=hard_col)

            except:
                self.transmission_data = pd.DataFrame(data)
                print('TRANSMISSION: Check the data or the columns')
                print(columns)

    # Get the 's DATA ===============================================================================================================================

    def _get_transformer_data(self):      

        # columns = self.lines[self.s-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data_raw, data = [], []

        head = [self.lines[i].strip() for i in range(self.f_trs, self.l_trs + 1) if '/' in self.lines[i].strip()]


        for line in head:
            trs_info = line.strip().replace('/', ' ').replace('\'', ' ').replace(' ', '').split(',')
            data.append(trs_info)

        # for i in range(len(data_raw)//4):
        #     prov = data_raw[i*4]

        #     # for d in data_raw[i*2+1]:
        #     #     prov.append(d)

        #     data.append(prov)

        try:

            hard_col = ['BUS1', 'BUS2', 'BUS3', 'CI', 'G_MAG', 'B_MAG', 'WI1', 'WI2', 'WI3', 'WI_V', 'WI_A', 'ARE', 'OWN', 'NAME']#, 'BUS4',  'NAME2']
                        # 'TYPE', 'RP', 'XP', 'TAP', 'PHASE', 'LIM1', 'LIM2', 'LIM3', 'CTR_STA', 'CTR_BUS', 'REM_CTR',
                        # 'UP_TAP', 'LO_TAP', 'UP_VOL', 'LO_VOL', 'TAP_STEP', 'NCT', 'CONC_TYPE', 'RN', 'XN', 'ZE_G', 'ZE_B', 'TO_G', 'TO_B']
            self.transformer_data = pd.DataFrame(data, columns=hard_col)

        except:
            self.transformer_data = pd.DataFrame(data)
            print('TRANSFORMER: Check the data or the columns')
            print(hard_col)

    # Get the GENERATION's DATA ===============================================================================================================================

    def concat(self):

        self.data = self.bus_data.merge(self.load_data, on='BUS_ID', how='outer')
        self.data = self.data.merge(self.gen_data, on='BUS_ID', how='left')

    # Get the INFORMATION  ===============================================================================================================================

    def networkInfo(self, show=False):

        self.total_PL_MW   = self.load_data['PL_MW'].sum()   
        self.total_QL_MVAR = self.load_data['QL_MVAR'].sum() 
        # print(self.total_PL_MW, self.total_QL_MVAR)

        self.total_PG_MW   = self.gen_data['PG_MW'].sum()    
        self.total_QG_MVAR = self.gen_data['QG_MVAR'].sum() 
        # print(self.total_PG_MW, self.total_QG_MVAR)

        self.total_PMAX_MW = self.gen_data['PMAX_MW'].sum()  
        self.total_PMIN_MW = self.gen_data['PMIN_MW'].sum()
        # print(self.total_PMAX_MW, self.total_PMIN_MW)

        self.reserva_total   = self.total_PMAX_MW - self.total_PG_MW
        self.reserva_per_gen = self.gen_data['PMAX_MW'] - self.gen_data['PG_MW']

        self.total_PG_RENEW = self.gen_data[self.gen_data['TYPE'] == 4]['PG_MW'].sum() 
        self.total_PG_SYNC  = self.gen_data[(self.gen_data['BLOCKED'] == 0)]['PG_MW'].sum() 
        # print(self.total_PG_RENEW, self.total_PG_SYNC)

        self.total_PMAX_MW_unblock  = self.gen_data[self.gen_data['BLOCKED'] == 0]['PMAX_MW'].sum() 
        self.total_PMAX_MW_block    = self.gen_data[self.gen_data['BLOCKED'] == 1]['PMAX_MW'].sum() 

        self.total_PG_MW_unblock    = self.gen_data[self.gen_data['BLOCKED'] == 0]['PG_MW'].sum() 
        self.total_PG_MW_block      = self.gen_data[self.gen_data['BLOCKED'] == 1]['PG_MW'].sum() 
        
        self.total_headroom_unblock = self.total_PMAX_MW_unblock - self.total_PG_MW_unblock
        
        
        self.total_headroom        = self.total_PMAX_MW - self.total_PG_MW
        # print('UNBLOCKED:', self.total_PMAX_MW_unblock)
        
        self.penetration = self.total_PG_RENEW / self.total_PG_MW
        
        if show:
            
            print(' ====================== Network Info ====================== ')
            print(' ======================== GENERATOR ======================= \n')
            print(f' Total Active Power:   {self.total_PG_MW:.4f} MW')
            print(f' Total Reactive Power: {self.total_QG_MVAR:10.4f} MVar\n')
            
            print(f' Total Max Active Power: {self.total_PMAX_MW:.4f} MW')
            print(f' Total Min Active Power: {self.total_PMIN_MW:10.4f} MW\n')
            
            print(f' Total Renew Active Power: {self.total_PG_RENEW:10.4f} MW')
            print(f' Total Sync Active Power:  {self.total_PG_SYNC:.4f} MW\n')
            
            print(f' Total Headroom:             {self.total_headroom:10.4f} MW')
            print(f' Total Max Unblocked Power:  {self.total_PMAX_MW_unblock:.4f} MW')
            print(f' Total Head Unblocked Power: {self.total_headroom_unblock:10.4f} MW\n')
            
            print(f' Renewable Penetration: {self.penetration*100:.4f} %\n')

            print(f' Total Max Unblocked Power:  {self.total_PMAX_MW_unblock:.4f} MW')
            print(f' Total Max Blocked Power:    {self.total_PMAX_MW_block:10.4f} MW\n')

            print(f' Total Unblocked Power:  {self.total_PG_MW_unblock:10.4f} MW')
            print(f' Total Blocked Power:    {self.total_PG_MW_block:10.4f} MW')
            
            print(' ========================== LOAD ========================== \n')
            print(f' Total Active Power:   {self.total_PL_MW:10.4f} MW')
            print(f' Total Reactive Power: {self.total_QL_MVAR:10.4f} MVar\n\n')


if __name__ == '__main__':

    path = 'Data/68bus.ntw'

    ND = NTW_Reader(path)
    ND.concat()


    # print(ND.transformer_data)

    # ND.bus_data.to_excel("bus_data.xlsx") 
    # ND.load_data.to_excel("load_data.xlsx")
    # ND.gen_data.to_excel("gen_data.xlsx")

    ND.transmission_data.to_csv("68bus_transmission_data.csv")
    ND.transformer_data.to_csv("68bus_transformer_data.csv")  
    ND.data.to_csv("68bus_data.csv")

