from RST_Reader import *
import pandas as pd
import numpy as np


class RST_Plot():

    def __init__(self, repots_path, contigences_path, eol=None, sol=None):

        self.data = RST_Process(repots_path, contigences_path).data

        print(len(self.data['Operational Point'].unique()))
        
        self.data['Day']     = [OP.split('/')[-2].replace('Dia', "") for OP in self.data['Operational Point'].values]
        self.data['Day_int'] = self.data['Day'].astype('int')
        self.data['Hour']    = [OP.split('/')[-1].split('_')[-1] for OP in self.data['Operational Point'].values]
        # self.data['key']     = self.data['Operational Point'] + self.data['Contigence']

        # INSTABILIDADE

        keys_inst = self.data[(self.data['A'] == 1.0) & (self.data['SIGLA'] == 'STAB')]['key'].unique()

        self.data_i = self.data[self.data['key'].isin(keys_inst)].reset_index(drop=True)
        self.data_e = self.data[~self.data['key'].isin(keys_inst)].reset_index(drop=True)

        # CÓDIGOS

        keys_code = self.data[(self.data['A'] != 0.0) & (self.data['SIGLA'] == 'CODE')]['key'].unique()

        self.data_c = self.data[self.data['key'].isin(keys_code)].reset_index(drop=True)
        self.data_n = self.data[~self.data['key'].isin(keys_code)].reset_index(drop=True)

        # COLORS

        colors    = ['lightsteelblue', 'royalblue', 'lightgreen', 'green', 'tan', 'darkgoldenrod', 'thistle', 'purple', 'lightcoral', 'red']
        cmap_name = 'my_list'
        self.cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

        # RENOVAVEIS

        if eol is not None:
            self.demanda_eol = pd.read_csv(eol, sep=';')

        if sol is not None:
            self.demanda_sol = pd.read_csv(sol, sep=';')

        if (sol is not None) and (eol is not None):
            self.renovaveis = self.demanda_eol.merge(self.demanda_sol, on=['periodo', 'hora'], how='outer')

            self.renovaveis['Day']  = [OP.split('/')[0]     for OP in self.renovaveis['periodo'].values]
            self.renovaveis['Hour'] = [OP.replace(':', '-') for OP in self.renovaveis['hora'].values]
            self.renovaveis['%MW']  = self.renovaveis[['MW_x', 'MW_y']].sum(axis=1)*100

            self.renovaveis['%MW_r']  = [round(mw, 0) for mw in self.renovaveis['%MW'].values]

            # print(self.renovaveis[['Day', 'Hour', '%MW']])

            self.data_n = self.data_n.merge(self.renovaveis[['Day', 'Hour', '%MW', '%MW_r']], on=['Day', 'Hour'], how='left')

    # ================================================================================================================================= #

    def scatter_hist(self, x, y, ax, ax_histx, ax_histy, n_bins=20):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        

        xmax, xmin = np.max(np.abs(x)), np.min(np.abs(x))
        binwidth = (xmax - xmin)/n_bins
        lim_max, lim_min = (int(xmax/binwidth) + 1) * binwidth, (int(xmin/binwidth)) * binwidth
        xbins = np.arange(lim_min, lim_max + binwidth, binwidth)
        ax_histx.hist(x, bins=xbins)

        ymax, ymin = np.max(np.abs(y)), np.min(np.abs(y))
        binwidth = (ymax - ymin)/n_bins
        lim_max, lim_min = (int(ymax/binwidth) + 1) * binwidth, (int(ymin/binwidth)) * binwidth
        ybins = np.arange(lim_min, lim_max + binwidth, binwidth)
        ax_histy.hist(y, bins=ybins, orientation='horizontal')

    def scatter_hist3d(self, x, y, z, ax, ax_histx, ax_histy, n_bins=20):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        points = ax.scatter(x, y, c=z, cmap=self.cmap)

        # now determine nice limits by hand:
        

        xmax, xmin = np.max(np.abs(x)), np.min(np.abs(x))
        binwidth = (xmax - xmin)/n_bins
        lim_max, lim_min = (int(xmax/binwidth) + 1) * binwidth, (int(xmin/binwidth)) * binwidth
        xbins = np.arange(lim_min, lim_max + binwidth, binwidth)
        ax_histx.hist(x, bins=xbins)

        ymax, ymin = np.max(np.abs(y)), np.min(np.abs(y))
        binwidth = (ymax - ymin)/n_bins
        lim_max, lim_min = (int(ymax/binwidth) + 1) * binwidth, (int(ymin/binwidth)) * binwidth
        ybins = np.arange(lim_min, lim_max + binwidth, binwidth)
        ax_histy.hist(y, bins=ybins, orientation='horizontal')

        return points

    # ================================================================================================================================= #

    def _fix(self, x, y, _x='A', _y='A'):
        
        xkey, ykey = x['key'].values, y['key'].values

        x = x[x['key'].isin(xkey)].reset_index(drop=True).rename(columns={_x:'x'})
        y = y[y['key'].isin(ykey)].reset_index(drop=True).rename(columns={_y:'y'})

        data = x[['key', 'x']].merge(y[['key', 'y']], on='key', how='inner')

        return data['x'].values, data['y'].values

























''' 
*****************************************************************************************************************************************************************

    RST_Plot_instavel -> RST_Plot
    
*****************************************************************************************************************************************************************
''' 


class RST_Plot_instavel(RST_Plot):

    # ===================================================================================================================================================== #

    def plot_inst_days_hours(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB']
        inst = inst.groupby(['Day', 'Day_int', 'Hour'])['A'].sum().reset_index(drop=False)
        inst = inst.sort_values(by=['Hour', 'Day_int'])

        x, y, z = inst['Hour'], inst['Day_int'], inst['A']

        plt.figure(figsize=(14, 6))

        points = plt.scatter(x, y, c=z, s=110, cmap=self.cmap)

        plt.colorbar(points, ticks=[i*5 for i in range(1, 11)])
        plt.xticks(rotation=75)
        plt.yticks([i for i in range(1, 29)], [str(i) for i in range(1, 29)])
        plt.title('Numero de Contingências com Instabilidade Transitória')
        plt.xlabel('Hora')
        plt.ylabel('Dia')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_days_hours.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_contigence_bus(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB']
        inst = inst.astype({'Contigence_Number':'int'})
        inst = inst.groupby(['Contigence_Number', 'B'])['A'].sum().reset_index(drop=False)
        inst = inst.sort_values(by=['B', 'Contigence_Number'])

        x, y, z = inst['B'].astype('int').astype('str'), inst['Contigence_Number'], inst['A']

        plt.figure(figsize=(16, 8))

        points = plt.scatter(x, y, c=z, s=60, cmap=self.cmap)

        plt.colorbar(points, ticks=[i*50 for i in range(1, 11)])
        plt.xticks(rotation=90)
        plt.yticks([i for i in range(1, 52)], [str(i) for i in range(1, 52)])
        plt.title('Numero Contingências com Instabilidades Transitórias')
        plt.xlabel('Barra')
        plt.ylabel('Número da Contingência')

        plt.grid(axis='y')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_contigence_bus.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_contigence_op(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)
        test = inst.groupby(['Operational Point'])['A'].sum().reset_index(drop=False)
        test = test.sort_values(by='A')
        test = test[test['A'] > 25]
        inst = inst[inst['Operational Point'].isin(test['Operational Point'].values)]

        inst['contInt'] = inst['Contigence_Number'].astype('int')
        inst = inst.sort_values(by=['contInt', 'Day_int'])

        inst['Nome'] = inst['Day'] + ' - ' + inst['Hour']

        x, y = inst['contInt'], inst['Nome']

        plt.figure(figsize=(14, 6))
        plt.grid()

        plt.scatter(x, y, s=100)

        plt.xticks([i for i in range(1, 52)], [str(i) for i in range(1, 52)], rotation=90)
        plt.title('Pontos de Operação com Instabilidades Transitórias\nPontos de Operação com mais de 25 contingências com instabilidade')
        plt.xlabel('Número da Contingência')
        plt.ylabel('Ponto de Operação')
        
        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_contigence_op.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_contingence(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)
        inst = inst.groupby(['Contigence_Number'])['A'].sum().reset_index(drop=False)
        inst['contInt'] = inst['Contigence_Number'].astype('int')
        inst = inst.sort_values(by='contInt')
        inst = inst[inst['A'] > 25]

        x, y = inst['Contigence_Number'], inst['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.ylabel('Número de Pontos de Operação com Instabilidade')
        plt.xlabel('Contingência')
        plt.title('Número Contingências com Instabilidades Transitórias\nContingências com mais de 25 POs com instabilidade')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_contingence.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_operation_points(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)

        inst['Nome'] = inst['Day'] + ' - ' + inst['Hour']

        inst = inst.groupby(['Nome'])['A'].sum().reset_index(drop=False)
        inst = inst.sort_values(by='Nome')
        inst = inst[inst['A'] > 25]

        

        x, y = inst['Nome'], inst['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.barh(x, y)
        plt.bar_label(bar_container)

        plt.xticks(rotation=0)
        plt.ylabel('Ponto de Operação')
        plt.xlabel('Número de Contingências com Instabilidade')
        plt.title('Número Contingências com Instabilidades Transitórias\nPOs com mais de 25 contingências com instabilidade')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_operation_points.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_day(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)
        inst = inst.groupby(['Day', 'Day_int'])['A'].sum().reset_index(drop=False)   
        inst = inst.sort_values(by='Day_int')
        inst = inst[inst['A'] > 15]

        x, y = inst['Day_int'], inst['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.xticks([i for i in range(1, 29)], [str(i) for i in range(1, 29)], rotation=0)
        plt.ylabel('Número de Contingências com Instabilidade')
        plt.xlabel('Dia')
        plt.title('Número Contingências com Instabilidades Transitórias\nDias com mais de 15 contingências com instabilidade')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_day.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_hour(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)
        inst = inst.groupby(['Hour'])['A'].sum().reset_index(drop=False)   
        inst = inst.sort_values(by='Hour')
        inst = inst[inst['A'] > 15]

        x, y = inst['Hour'], inst['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.xticks(rotation=90)
        plt.ylabel('Número de Contingências com Instabilidade')
        plt.xlabel('Hora')
        plt.title('Número Contingências com Instabilidades Transitórias\nHoras com mais de 15 contingências com instabilidade')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_hour.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_bus(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB'].reset_index(drop=True)
        inst = inst.groupby(['B'])['A'].sum().reset_index(drop=False)   
        inst = inst.sort_values(by='B')
        inst = inst[inst['A'] > 15]

        x, y = inst['B'].astype('int').astype('str'), inst['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.xticks(rotation=90)
        plt.ylabel('Número de Contingências com Instabilidade')
        plt.xlabel('Barra')
        plt.title('Número Contingências com Instabilidades Transitórias\nBarras associadas a mais de 15 contingências com instabilidade')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_bus.png', bbox_inches="tight")        
    
    # ===================================================================================================================================================== #

    def plot_inst_histogram_CODE(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'CODE'].reset_index(drop=True)

        print(len(inst))
        print(inst['A'].value_counts().reset_index(drop=False))

        codes = inst['A'].value_counts().reset_index(drop=False)
        codes = codes.sort_values(by='index', ascending=True)

        x, y = codes['index'].astype('int').astype('str'), codes['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.ylabel('Número de Apariações')
        plt.xlabel('Código')
        plt.title('Número de Apariações de cada Código')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_CODE.png', bbox_inches="tight")

     # ===================================================================================================================================================== #

    def plot_code_histogram_CODE(self):

        inst = self.data_c[self.data_c['SIGLA'] == 'CODE'].reset_index(drop=True)

        print(len(inst))
        print(inst['A'].value_counts().reset_index(drop=False))

        codes = inst['A'].value_counts().reset_index(drop=False)
        codes = codes.sort_values(by='index', ascending=True)

        x, y = codes['index'].astype('int').astype('str'), codes['A']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        plt.ylabel('Número de Apariações')
        plt.xlabel('Código')
        plt.title('Número de Apariações de cada Código')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_code_histogram_CODE.png', bbox_inches="tight")


























''' 
*****************************************************************************************************************************************************************

    RST_Plot_estavel -> RST_Plot

*****************************************************************************************************************************************************************
''' 


class RST_Plot_estavel(RST_Plot):

    # ===================================================================================================================================================== #

    def plot_est_violin_rocof(self):    

        bad_key = self.data[(self.data['A'] == 1)]['key'].unique()
        est_raw = self.data[(~self.data['key'].isin(bad_key))]

        _rocofs, _labels, _colors = [], [], []
        rocofs , labels ,  colors = [], [], []

        est_raw['Contigence_Number_int'] = est_raw['Contigence_Number'].astype('int')

        t = (est_raw['SIGLA'] == 'RCFC') & (est_raw['Contigence_Number'] != '10')

        print(est_raw[t][['Operational Point', 'Contigence', 'Contigence_Number', 'A']].sort_values(by='A', ascending=False).head(25))

        for idx, cont in enumerate(sorted(est_raw['Contigence_Number_int'].unique())):

            est    = est_raw[est_raw['Contigence_Number_int'] == cont]
            cor    = 'red' if any(((est['SIGLA'] == 'RCFC') & (est['A'] > 2.5))) else 'lightskyblue'
            filt_y = (est['SIGLA'] == 'RCFC') & (est['A'] < 2.5)

            _rocofs.append(est[filt_y]['A'].values)
            _labels.append(cont)
            _colors.append(cor)            

            if (idx+1)%9 == 0 or idx+1 == len(est_raw['Contigence_Number'].unique()):
                rocofs.append(_rocofs)
                labels.append(_labels)
                colors.append(_colors)
                _rocofs, _labels, _colors = [], [], []

        

        fig, axs = plt.subplots(2, 3, figsize=(30, 13), sharey=True)
        x, y     = 0, 0 
        for r, l, c in zip(rocofs, labels, colors):

            vio = axs[y, x].violinplot(r, showmeans=True, showmedians=True)

            vio['cmeans'].set_color(['red' for i in range(len(r))])
            for pc, color in zip(vio['bodies'], c): pc.set_facecolor(color)

            axs[y, x].set_xticks([i for i in range(1, len(l)+1)], l, rotation=45)

            fig.suptitle('Gráfico Violino: RoCoF x Contingência')

            if x == 0: axs[y, x].set_ylabel('Distrubuição do RoCoF')
            if y == 1: axs[y, x].set_xlabel('Contingências')

            axs[y, x].grid(True, axis='y')

            
            # axs[y, x].set_yticklabels(fontsize=20)

            y = 1 if x == 2 else y
            x = 0 if x == 2 else x+1

        # plt.tick_params(axis='x', fontsize=20)
        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_violin_rocof.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_est_violin_nadir(self):

        est_raw = self.data_e
        est_raw.loc[est_raw['SIGLA'] == 'NDRC', 'Nadir'] = 60 - est_raw[est_raw['SIGLA'] == 'NDRC']['A']

        _rocofs, _labels, _colors = [], [], []
        rocofs , labels ,  colors = [], [], []

        t = (est_raw['SIGLA'] == 'NDRC')
        est_raw['Nadir'] = 60  -est_raw['A']

        print(est_raw[t][['Operational Point', 'Contigence', 'Contigence_Number', 'Nadir']].sort_values(by='Nadir', ascending=True).head(25))

        for idx, cont in enumerate(est_raw['Contigence_Number'].unique()):

            est    = est_raw[est_raw['Contigence_Number'] == cont]
            cor = 'red' if any(((est['SIGLA'] == 'NDRC') & (est['Nadir'] < 58))) else 'lightskyblue'
            filt_y = (est['SIGLA'] == 'NDRC') & (est['Nadir'] > 58)

            _rocofs.append(est[filt_y]['Nadir'].values)
            _labels.append(cont)
            _colors.append(cor)            

            if (idx+1)%9 == 0 or idx+1 == len(est_raw['Contigence_Number'].unique()):
                rocofs.append(_rocofs)
                labels.append(_labels)
                colors.append(_colors)
                _rocofs, _labels, _colors = [], [], []

        fig, axs = plt.subplots(2, 3, figsize=(30, 13), sharey=True)
        x, y     = 0, 0 
        for r, l, c in zip(rocofs, labels, colors):

            bp  = axs[y, x].violinplot(r, showmeans=True, showmedians=True)

            bp['cmeans'].set_color(['red' for i in range(len(r))])
            for pc, color in zip(bp['bodies'], c):
                pc.set_facecolor(color)

            axs[y, x].set_xticks([i for i in range(1, len(l)+1)], l, rotation=45)

            fig.suptitle('Gráfico Violino: Frequência de Nadir  x Contingência')

            if x == 0: axs[y, x].set_ylabel('Distrubuição da Frequência de Nadir')
            if y == 1: axs[y, x].set_xlabel('Contingências')

            axs[y, x].grid(True, axis='y')
            # axs[y, x].set_yticklabels(fontsize=20)

            y = 1 if x == 2 else y
            x = 0 if x == 2 else x+1

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_violin_nadir.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_est_violin_damping(self):

        bad_key = self.data[(self.data['A'] == 1)]['key'].unique()
        est_raw = self.data_e

        

        _rocofs, _labels, _colors = [], [], []
        rocofs , labels ,  colors = [], [], []

        est_raw['Contigence_Number_int'] = est_raw['Contigence_Number'].astype('int')

        t = (est_raw['SIGLA'] == 'DAMP') #& (est_raw['Contigence_Number'] != '10')

        print(est_raw[t][['Operational Point', 'Contigence', 'Contigence_Number', 'A']].sort_values(by='A', ascending=False).head(25))
        print(len(est_raw[(est_raw['SIGLA'] == 'DAMP')]))
        print(len(est_raw[(est_raw['SIGLA'] == 'DAMP') & (est_raw['A'] > 0)]))
        print(len(est_raw[(est_raw['SIGLA'] == 'DAMP') & (est_raw['A'] < 0)]))

        for idx, cont in enumerate(sorted(est_raw['Contigence_Number_int'].unique())):

            est    = est_raw[est_raw['Contigence_Number_int'] == cont]

            if any(((est['SIGLA'] == 'DAMP') & (est['A'] < -10) & (est['A'] > 10))):
                
                cor = 'gray'

            elif any(((est['SIGLA'] == 'DAMP') & (est['A'] < -10))):
                
                cor = 'red'

            elif any(((est['SIGLA'] == 'DAMP') & (est['A'] > 10))):

                cor = 'green'
            
            else:

                cor = 'lightskyblue'

            filt_y = (est['SIGLA'] == 'DAMP') & (est['A'] > -10) & (est['A'] < 10)

            _rocofs.append(est[filt_y]['A'].values)
            _labels.append(cont)
            _colors.append(cor)        

            if (idx+1)%9 == 0 or idx+1 == len(est_raw['Contigence_Number'].unique()):
                rocofs.append(_rocofs)
                labels.append(_labels)
                colors.append(_colors)
                _rocofs, _labels, _colors = [], [], []

        fig, axs = plt.subplots(2, 3, figsize=(30, 13), sharey=True)
        x, y     = 0, 0 
        for r, l, c in zip(rocofs, labels, colors):

            bp  = axs[y, x].violinplot(r, showmeans=True, showmedians=True)

            bp['cmeans'].set_color(['red' for i in range(len(r))])
            for pc, color in zip(bp['bodies'], c):
                pc.set_facecolor(color)

            axs[y, x].set_xticks([i for i in range(1, len(l)+1)], l, rotation=45)

            fig.suptitle('Gráfico Violino: Amortecimento  x Contingência')

            if x == 0: axs[y, x].set_ylabel('Distrubuição da Amortecimento')
            if y == 1: axs[y, x].set_xlabel('Contingências')

            axs[y, x].grid(True, axis='y')
            # axs[y, x].set_yticklabels(fontsize=20)

            y = 1 if x == 2 else y
            x = 0 if x == 2 else x+1

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_violin_damping.png', bbox_inches="tight")   

    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_PGTM_NDRC_NDRC(self):


        a = self.data_n[(self.data_n['SIGLA'] == 'NDRC') & (self.data_n['C'] > 10)]['key'] # & (self.data_n['Contigence_Number'] == '5')
        est_raw = self.data_e#[~(self.data_e['Contigence_Number'] == '6')]#.iloc[:5000]

        filt_x = est_raw['CODE'] == 'NDRC'
        filt_y = est_raw['CODE'] == 'NDRC'
        filt_z = (est_raw['CODE'] == 'RCFC') & (est_raw['A'] < 2.5)

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='B')
        x, z = self._fix(est_raw[filt_x], est_raw[filt_z], _x='B')
        z, y = self._fix(est_raw[filt_z], est_raw[filt_y])
        # z_, y = self._fix(est_raw[filt_z], est_raw[filt_y], _x='D')

        print(len(x), len(y), len(z))


        # Start with a square Figure.
        fig = plt.figure(figsize=(10, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        points = self.scatter_hist3d(x, 60-y, z, ax, ax_histx, ax_histy)

        # plt.colorbar(points, location='left') #, orientation='horizontal'

        ax.set_ylabel('Frequência de Nadir')
        ax.set_xlabel('Tempo em que ocorre Frequência de Nadir')

        plt.suptitle('Frequência de Nadir x Tempo de Ocorrência')

        plt.show()

    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_RCFC_NDRC(self):

        est_raw = self.data_e#[~(self.data_e['Contigence_Number'] == '10')]
        # est_raw = est_raw[~(est_raw['Contigence_Number'] == '3')]
        # est_raw = est_raw[~(est_raw['Contigence_Number'] == '4')] #& (self.data_e['Contigence_Number'] == '9')

        filt_x = ((est_raw['SIGLA'] == 'NDRC') & (est_raw['C'] > 0))
        filt_y = est_raw['SIGLA'] == 'RCFC'#) #( & (est_raw['A'] < 2.5)

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='C')

        print(est_raw[filt_y].sort_values(by='A', ascending=False).head(25))


        # Start with a square Figure.
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        self.scatter_hist(x, y, ax, ax_histx, ax_histy)

        ax.set_ylabel('RoCoF [Hz/s]')
        ax.set_xlabel('Inércia [MW/s]')
        plt.suptitle('RoCoF x Inércia')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_duplo_hist_RCFC_NDRC.png', bbox_inches="tight")

        # plt.show()


    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_NDRC_NDRC(self):

        est_raw = self.data_e#[~(self.data_e['Contigence_Number'] == '6') ]#.iloc[:5000] #& (self.data_e['Contigence_Number'] == '9')

        filt_x = ((est_raw['SIGLA'] == 'NDRC') & (est_raw['C'] > 0))
        filt_y = est_raw['SIGLA'] == 'NDRC'

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='C')

        a = est_raw[filt_y].sort_values(by='A', ascending=False).head(25)
        a['A'] = 60 - a['A']

        print(a)


        # Start with a square Figure.
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        self.scatter_hist(x, 60-y, ax, ax_histx, ax_histy)

        ax.set_ylabel('Frequência de Nadir [Hz]')
        ax.set_xlabel('Inércia [MW/s]')
        plt.suptitle('Frequência de Nadir x Inércia')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_duplo_hist_NDRC_NDRC.png', bbox_inches="tight")


    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_DAMP_NDRC(self):

        est_raw = self.data_e#[~(self.data_e['Contigence_Number'] == '10')]
        # est_raw = est_raw[~(est_raw['Contigence_Number'] == '3')]
        # est_raw = est_raw[~(est_raw['Contigence_Number'] == '4')] #& (self.data_e['Contigence_Number'] == '9')

        filt_x = ((est_raw['SIGLA'] == 'NDRC') & (est_raw['C'] > 0))
        filt_y = ((est_raw['SIGLA'] == 'DAMP') & (est_raw['A'] < 10) & (est_raw['A'] > -10))

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='C')

        print(est_raw[filt_y].sort_values(by='A', ascending=False).head(25))


        # Start with a square Figure.
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        self.scatter_hist(x, y, ax, ax_histx, ax_histy)

        ax.set_ylabel('Amortecimento')
        ax.set_xlabel('Inércia [MW/s]')
        plt.suptitle('Amortecimento x Inércia')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_est_duplo_hist_DAMP_NDRC.png', bbox_inches="tight")

    # ===================================================================================================================================================== #

    def plot_inst_histogram_bus_DAMP(self):

        est = self.data_e[self.data_e['SIGLA'] == 'DAMP'].reset_index(drop=True)

        est = est['B'].value_counts(dropna=False).reset_index(drop=False)
        est = est.sort_values(by='index')
        est = est[est['B'] > 600]

        print(est)

        # inst = inst.groupby(['Day', 'Day_int'])['A'].sum().reset_index(drop=False)   
        # inst = inst.sort_values(by='Day_int')
        # inst = inst[inst['A'] > 15]

        x, y = est['index'].astype('int').astype('str'), est['B']

        plt.figure(figsize=(14, 8))

        bar_container = plt.bar(x, y)
        plt.bar_label(bar_container)

        # plt.xticks([i for i in est['index'].unique()], [str(i) for i in est['index'].unique()], rotation=0)
        plt.ylabel('Frequência de Ocorrência')
        plt.xlabel('Barra')
        plt.title('Histograma da Frequência de Ocorrência das Barras (Amortecimento) \nFrquência > 600')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/plot_inst_histogram_bus_DAMP.png', bbox_inches="tight")





    def teste(self):

        est_raw = self.data_e[(self.data_e['Contigence_Number'] == '50')]#.iloc[:5000] #& (self.data_e['Contigence_Number'] == '9')

        filt_x = ((est_raw['SIGLA'] == 'NDRC') & (est_raw['C'] > 0))
        filt_y = est_raw['SIGLA'] == 'NDRC'

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='C')

        a = est_raw[filt_y].sort_values(by='A', ascending=False).head(25)
        a['A'] = 60 - a['A']

        print(a)


        # Start with a square Figure.
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        self.scatter_hist(x, 60-y, ax, ax_histx, ax_histy)

        ax.set_ylabel('Frequência de Nadir [Hz]')
        ax.set_xlabel('Inércia [MW/s]')
        plt.suptitle('Frequência de Nadir x Inércia')

        plt.legend(loc='best', bbox_to_anchor=(1, 1.1))
        plt.savefig('images/teste.png', bbox_inches="tight")




if __name__ == '__main__':

    ### INSTAVEL

    # RP = RST_Plot_instavel(repots_path='REV2.json',
    #                        contigences_path='contigences2.json',
    #                        eol='pu_EOL_da_demanda_bruta_SIN (1).csv',
    #                        sol='pu_SOL_da_demanda_bruta_SIN (1).csv')
    
    # RP.plot_inst_days_hours()
    # RP.plot_inst_contigence_bus()
    # RP.plot_inst_contigence_op()
    # RP.plot_inst_histogram_contingence()
    # RP.plot_inst_histogram_operation_points()
    # RP.plot_inst_histogram_day()
    # RP.plot_inst_histogram_hour()
    # RP.plot_inst_histogram_bus()
    # RP.plot_inst_histogram_CODE()
    # RP.plot_code_histogram_CODE()

    ### ESTAVEL

    RP = RST_Plot_estavel(repots_path='REV2.json',
                           contigences_path='contigences2.json',
                           eol='pu_EOL_da_demanda_bruta_SIN (1).csv',
                           sol='pu_SOL_da_demanda_bruta_SIN (1).csv')

    # RP.plot_est_violin_rocof()
    # RP.plot_est_violin_nadir()
    # RP.plot_est_violin_damping()
    # RP.plot_est_duplo_hist_RCFC_NDRC()
    # RP.plot_est_duplo_hist_NDRC_NDRC()
    # RP.plot_est_duplo_hist_DAMP_NDRC()
    RP.plot_inst_histogram_bus_DAMP()