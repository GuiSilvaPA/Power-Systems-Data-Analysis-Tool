import pandas as pd
import json, os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

class RST_Reader():

    def __init__(self, path):

        days = os.listdir(path)

        reports = []
        for day in days:
            hours = os.listdir(path + '/' + day)

            for hour in hours:
                reports.append(path + '/' + day + '/' + hour)

        self.reports = reports
        self.path    = path

    def generate_json(self, save_path):

        all_reports = {}
        for report_file in tqdm(self.reports):

            with open(report_file) as f:
                raw_lines = f.readlines()

            lines = [line.strip().split() for line in raw_lines]

            contigences, reports, cont = [], [], True
            for line in lines[3:]:

                try:
                    int(line[0])
                    if cont: contigences.append(line)
                    else: reports.append(line)

                except ValueError:
                    cont = False
                    pass

            # code_names = set([report[1] for report in reports])

            dict_report, actual_contigence, cont = {}, '0', 0
<<<<<<< HEAD
            print(report_file)
=======
>>>>>>> a84053ba715c15222ecdbc75461bca21e62f77c1
            for report in reports:

                for idx, var in enumerate(report[2:]):
                    report[idx+2] = float(var)


                if report[0] != actual_contigence:
                    cont = 0
                    actual_contigence = report[0]
                    dict_report['Contigence_' + actual_contigence] = {'Line_' + str(cont) : report}

                else:
                    cont += 1
                    dict_report['Contigence_' + actual_contigence]['Line_' + str(cont)] = report

            all_reports[report_file.split('.')[0].replace('Month/DS202210', '').replace('/20230406_C_', '/')] = dict_report


        with open(save_path, "w") as write_file:
            json.dump(all_reports, write_file, indent=4)

        with open('contigences2.json', "w") as write_file:
            json.dump(contigences, write_file, indent=4)




class RST_Process():
    def __init__(self, repots_path, contigences_path):

        with open(repots_path, 'r') as f:
            data = json.load(f)

        with open(contigences_path, 'r') as f:
            contigences = json.load(f)


        stab_points = []
        for operation_point in tqdm(data.keys()):
            for contigence in data[operation_point].keys():
                for params in data[operation_point][contigence]:

                    actual_param, param = [operation_point, contigence], data[operation_point][contigence][params]
                    for i in param: actual_param.append(i)

                    stab_points.append(actual_param)

        cont_number = ['Contigence_' + contigence[0] for contigence in contigences]
        cont_name   = [' '.join(contigence[1:]) for contigence in contigences]

        df = pd.DataFrame(stab_points, columns=['Operational Point', 'Contigence', 'Contigence_Number', 'SIGLA', 'A', 'B', 'C', 'D', 'E'])
        df = df.replace(cont_number, cont_name)

        df['Contigence'] = df.astype({'Contigence_Number':'str'})['Contigence_Number'] + '_' + df['Contigence']
        df['key'] = df['Operational Point'] + df['Contigence']

        self.data = df

        # print(df.head(5))




if __name__ == '__main__':

<<<<<<< HEAD
    path = 'rev2'
    RR = RST_Reader(path)
    print(RR.reports)
    RR.generate_json(save_path="REV2.json")
=======
    path = 'RST_V2A2F2_rev2_FluxoHppa_dynHPPA_PRM01'
    RR = RST_Reader(path)
    print(RR.reports)
    RR.generate_json(save_path="REV2-Nao_Oficial.json")
>>>>>>> a84053ba715c15222ecdbc75461bca21e62f77c1
