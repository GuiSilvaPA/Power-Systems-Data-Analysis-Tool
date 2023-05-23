from DataTool import *

PATH_NTW = 'Data/9bus.ntw'

ND = NTW_Reader(PATH_NTW)
ND.networkInfo()

porcentagens               = [(i*2)for i in range(5, 49)]
patamares_de_carga         = [(i*0.02)*ND.total_PMAX_MW for i in range(5, 49)]
multiplicadores_de_carga   = [carga/ND.total_PL_MW for carga in patamares_de_carga]

### GERANDO CENARIOS

for multiplicador_carga, porcentagem in zip(multiplicadores_de_carga, porcentagens):

    ND = NTW_Editor(PATH_NTW)
    ND.change_load(param='PL_MW', multi=multiplicador_carga, spec=None, arre=2, keepFP=True)
    
    ND.gen_data['PG_MW'] = ND.gen_data['PG_MW']*multiplicador_carga
    
    
    ND.save(save_path='Casos/' + PATH_NTW.split('/')[-1].split('.')[0] + '_' + str(porcentagem) + '.ntw')

### GERANDO CONTINGÊNCIAS

ED = EventData(time=30)

geradores = ND.DF_gen['BUS_ID'].values
names     = []

for idx, gen in enumerate(geradores):

    ED.new_event(name='GenLoss_' + str(gen), evento=18, info1=gen, param1=1, time=1, info2=0, info3=0, param2=0, param3=0)
    names.append('GenLoss_' + str(gen))
    
    if idx != len(geradores)-1:
        for gen_2 in geradores[idx+1:]:
            
            ED.new_event(name='GenLoss_' + str(gen) + '_' + str(gen_2), evento=18, info1=gen, param1=1, time=1, info2=0, info3=0, param2=0, param3=0)
            ED.append(n_event=len(ED.events), evento=18, info1=gen_2, param1=1, time=1.5, info2=0, info3=0, param2=0, param3=0)
            
            names.append('GenLoss_' + str(gen) + '_' + str(gen_2))