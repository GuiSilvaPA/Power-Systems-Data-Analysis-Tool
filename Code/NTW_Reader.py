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

        self._get_bus_data()
        self._get_load_data()
        self._get_gen_data()

    # Get the BUS's DATA ======================================================================================================================================

    def _get_bus_data(self):      

        columns = self.lines[self.f_bus-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_bus, self.l_bus + 1):
            bus_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(',', ' ').split()
            data.append(bus_info)

        try:
            self.bus_data = pd.DataFrame(data, columns=columns)

        except:
            self.bus_data = pd.DataFrame(data)
            print('BUS: Check the data or the columns')
            print(columns)

    # Get the LOAD's DATA =====================================================================================================================================

    def _get_load_data(self):      

        columns = self.lines[self.f_load-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_load, self.l_load + 1):
            load_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(',', ' ').split()
            data.append(load_info)

        try:
            self.load_data = pd.DataFrame(data, columns=columns)

        except:
            self.load_data = pd.DataFrame(data)
            print('LOAD: Check the data or the columns')

    # Get the GENERATION's DATA ===============================================================================================================================

    def _get_gen_data(self):      

        columns = self.lines[self.f_gen-1].strip().replace('/', ' ').replace('(', ' ').replace(')', ' ').replace('\'', ' ').replace(',', ' ').split()
        data = []

        for i in range(self.f_gen, self.l_gen + 1):
            gen_info = self.lines[i].strip().replace('/', ' ').replace('\'', ' ').replace(',', ' ').split()
            data.append(gen_info)

        try:
            self.gen_data = pd.DataFrame(data, columns=columns)

        except:
            self.gen_data = pd.DataFrame(data)
            print('GEN: Check the data or the columns')

    # Get the GENERATION's DATA ===============================================================================================================================

    def concat(self):

        self.data = self.bus_data.merge(self.load_data, on='BUS_ID', how='left')
        self.data = self.data.merge(self.gen_data, on='BUS_ID', how='left')


if __name__ == '__main__':

    path = 'DataForTest/SIN.ntw'

    ND = NTW_Reader(path)

    ND.bus_data.to_excel("bus_data.xlsx") 
    ND.load_data.to_excel("load_data.xlsx")
    ND.gen_data.to_excel("gen_data.xlsx")

    ND.concat()
    ND.data.to_excel("data.xlsx")

