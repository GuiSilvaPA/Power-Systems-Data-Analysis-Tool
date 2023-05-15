from NTW_Reader import NTW_Reader



class NTW_Editor(NTW_Reader):

    def change_bus(self):

        print(self.bus_data)


    def change_gen(self):

        print(self.gen_data.columns)

    
    def change_load(self):

        print(self.gen_data)







if __name__ == '__main__':

    path = 'C:/Users/albing-local/Desktop/Data/DataForTest/9bus.ntw'

    NE = NTW_Editor(path)
    NE.change_gen()