from RST_Reader import *
import pandas as pd
import numpy as np


class RST_Plot():

    def __init__(self, repots_path, contigences_path, eol=None, sol=None):

        self.data = RST_Process(repots_path, contigences_path).data
        
        self.data['Day']     = [OP.split('/')[-2].replace('DS202210', "") for OP in self.data['Operational Point'].values]
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

    def plot_inst_days_hours(self):

        inst = self.data_i[self.data_i['SIGLA'] == 'STAB']
        inst = inst.groupby(['Day', 'Day_int', 'Hour'])['A'].sum().reset_index(drop=False)

        print(inst.sort_values(by='A').head(10))

        inst = inst.sort_values(by=['Hour'])

        x, y, z = inst['Hour'], inst['Day_int'], inst['A']

        plt.figure(figsize=(14, 6))

        points = plt.scatter(x, y, c=z, s=50, cmap=self.cmap)

        plt.colorbar(points, ticks=[i*5 for i in range(1, 11)])
        plt.xticks(rotation=75)
        plt.yticks(ticks=[int(i) for i in range(1, 29)])
        plt.title('Numero de Contingências com Instabilidade Transitória')
        plt.xlabel('Hora')
        plt.ylabel('Dia')

        plt.show()



''' 
*****************************************************************************************************************************************************************

    RST_Plot_estavel -> RST_Plot

*****************************************************************************************************************************************************************
''' 


class RST_Plot_estavel(RST_Plot):

    # ================================================================================================================================= #

    def plot_est_violin_rocof(self):

        bad_key = self.data[(self.data['A'] == 1)]['key'].unique()
        est_raw = self.data[(~self.data['key'].isin(bad_key))]

        self.data_n

        _rocofs, _labels, _colors = [], [], []
        rocofs , labels ,  colors = [], [], []
        for idx, cont in enumerate(est_raw['Contigence_Number'].unique()):

            est    = est_raw[est_raw['Contigence_Number'] == cont]
            cor = 'red' if any(((est['CODE'] == 'RCFC') & (est['A'] > 2.5))) else 'lightskyblue'
            filt_y = (est['CODE'] == 'RCFC') & (est['A'] < 2.5)

            _rocofs.append(est[filt_y]['A'].values)
            _labels.append(cont)
            _colors.append(cor)            

            if (idx+1)%10 == 0 or idx+1 == len(est_raw['Contigence_Number'].unique()):
                rocofs.append(_rocofs)
                labels.append(_labels)
                colors.append(_colors)
                _rocofs, _labels, _colors = [], [], []

        for r, l, c in zip(rocofs, labels, colors):

            fig = plt.figure()
            bp  = plt.violinplot(r, showmeans=True, showmedians=True)

            bp['cmeans'].set_color(['red' for i in range(len(r))])
            for pc, color in zip(bp['bodies'], c):
                pc.set_facecolor(color)

            plt.xticks([i for i in range(1, len(l)+1)], l, rotation=45)

            plt.ylabel('Distributed RoCoF')
            plt.xlabel('Contigences')
            plt.title('Violin Plot: RoCoF x Contigence')

            plt.show()


    # ================================================================================================================================= #

    def plot_est_violin_rocof(self):

        bad_key = self.data[(self.data['A'] == 1)]['key'].unique()
        est_raw = self.data[(~self.data['key'].isin(bad_key))]

        self.data_n

        _rocofs, _labels, _colors = [], [], []
        rocofs , labels ,  colors = [], [], []
        for idx, cont in enumerate(est_raw['Contigence_Number'].unique()):

            est    = est_raw[est_raw['Contigence_Number'] == cont]
            cor = 'red' if any(((est['CODE'] == 'RCFC') & (est['A'] > 2.5))) else 'lightskyblue'
            filt_y = (est['CODE'] == 'RCFC') & (est['A'] < 2.5)

            _rocofs.append(est[filt_y]['A'].values)
            _labels.append(cont)
            _colors.append(cor)            

            if (idx+1)%10 == 0 or idx+1 == len(est_raw['Contigence_Number'].unique()):
                rocofs.append(_rocofs)
                labels.append(_labels)
                colors.append(_colors)
                _rocofs, _labels, _colors = [], [], []

        for r, l, c in zip(rocofs, labels, colors):

            fig = plt.figure()
            bp  = plt.violinplot(r, showmeans=True, showmedians=True)

            bp['cmeans'].set_color(['red' for i in range(len(r))])
            for pc, color in zip(bp['bodies'], c):
                pc.set_facecolor(color)

            plt.xticks([i for i in range(1, len(l)+1)], l, rotation=45)

            plt.ylabel('Distributed RoCoF')
            plt.xlabel('Contigences')
            plt.title('Violin Plot: RoCoF x Contigence')

            plt.show()















    def rocof_COI_X_penetracao(self):

        data = self.data_n[(self.data_n['SIGLA'] == 'RCFC') & (self.data_n['A'] < 3)] #  

        print(data)

        for cont in range(1, 52):

            temp    = data[(data['Contigence_Number'] == str(cont))]
            x, y, z = temp['%MW_r'], temp['A'], None

            plt.scatter(x, y, c=z, label='Cont. ' + str(cont)) #, c=z, s=50, cmap=self.cmap

            # break

            plt.ylabel('RoCoF (COI) [Hz/s]')
            plt.xlabel('Penetração [% MW]')
            plt.legend()

            plt.show()
   

    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_RCFC_NDRC(self):

        data = self.data_n[(self.data_n['SIGLA'] == 'NDRC') & (self.data_n['C'] > 10)]

        x, y = data['C'], data['%MW']

        fig = plt.figure(figsize=(10, 6))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

        ax       = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)


        self.scatter_hist(x, y, ax, ax_histx, ax_histy)

        plt.show()




    # ===================================================================================================================================================== #

    def plot_est_duplo_hist_PGTM_NDRC_NDRC(self):

        a = self.data_n[(self.data_n['SIGLA'] == 'NDRC') & (self.data_n['C'] > 10) & (self.data_n['Contigence_Number'] == '5')]['key']

        est_raw = self.data_n[self.data_n['key'].isin(a)]

        filt_x = (est_raw['SIGLA'] == 'NDRC')
        filt_y = (est_raw['SIGLA'] == 'NDRC')
        filt_z = (est_raw['SIGLA'] == 'RCFC') & (est_raw['A'] < 1.5)

        x, y = self._fix(est_raw[filt_x], est_raw[filt_y], _x='C', _y='%MW')
        x, z = self._fix(est_raw[filt_x], est_raw[filt_z], _x='C')
        z, y = self._fix(est_raw[filt_z], est_raw[filt_y], _y='%MW')

        print(len(x), len(y), len(z))

        fig = plt.figure(figsize=(10, 6))
        gs  = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

        # Create the Axes.
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        # Draw the scatter plot and marginals.
        points = self.scatter_hist3d(x, y, z, ax, ax_histx, ax_histy)

        plt.colorbar(points, location='left') 

        plt.show()




if __name__ == '__main__':

    RP = RST_Plot_estavel(repots_path='REV2-Nao_Oficial.json',
                           contigences_path='Cont-REV2-Nao_Oficial.json',
                           eol='pu_EOL_da_demanda_bruta_SIN (1).csv',
                           sol='pu_SOL_da_demanda_bruta_SIN (1).csv')
    
    print(RP.data_n)
    
    RP.plot_est_duplo_hist_PGTM_NDRC_NDRC()

    # RP.plot_inst_days_hours()