import mne
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import signal
from matplotlib.pyplot import figure
import scipy.signal
import neurokit2 as nk
import plotly.express as px
from shapely.geometry import Polygon


def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def discrete_signal_resample(signal, time, new_sampling_rate):
    # Текущая частота дискретизации
    current_sampling_rate = 1 / np.mean(np.diff(time))

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * new_sampling_rate / current_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)
    new_time = np.linspace(time[0], time[-1], num_points_new)

    return new_signal, new_time


def calculate_area(points):
    polygon = Polygon(points)
    area_inside_loop = polygon.area
    return area_inside_loop


def loop(df_term, name, show=False):
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']

    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    if show:
        plt.figure(figsize=(15, 5), dpi=80)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.x,df_term.y)
        plt.title('Фронтальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 3, 2)
        plt.plot(df_term.y,df_term.z)
        plt.title('Сагитальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.x, df_term.z)
        plt.title('Аксиальная плоскость')  
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.suptitle(f'{name} петля', fontsize=16)
        plt.show()
    
    points = list(zip(df_term['x'], df_term['y']))
    area_inside_loop_1 = calculate_area(points)
    print(f"Площадь петли {name} во фронтальной плоскости:", area_inside_loop_1)

    points = list(zip(df_term['y'], df_term['z']))
    area_inside_loop_2 = calculate_area(points)
    print(f"Площадь петли {name} в сагитальной плоскости:", area_inside_loop_2)

    points = list(zip(df_term['x'], df_term['z']))
    area_inside_loop_3 = calculate_area(points)
    print(f"Площадь петли {name} в аксиальной плоскости:", area_inside_loop_3)

    return area_inside_loop_1, area_inside_loop_2, area_inside_loop_3


def get_area(show, df, waves_peak, start, Fs_new, QRS, T):
    area = []
    if QRS:
        closest_Q_peak = min(waves_peak['ECG_Q_Peaks'], key=lambda x: abs(x - start))
        closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
        df_new = df.copy()
        df_term = df_new.iloc[closest_Q_peak:closest_S_peak,:]
        df_row = df_new.iloc[closest_Q_peak:closest_Q_peak+1,:]
        df_term = pd.concat([df_term, df_row])

        area = list(loop(df_term, name='QRS', show=show))

    if T:
        closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
        closest_T_end = min(waves_peak['ECG_T_Offsets'], key=lambda x: abs(x - closest_S_peak))
        df_new = df.copy()
        df_term = df_new.iloc[closest_S_peak + int(0.025*Fs_new) : closest_T_end, :]
        df_row = df_new.iloc[closest_S_peak+int(0.025*Fs_new):closest_S_peak+int(0.025*Fs_new)+1,:]
        df_term = pd.concat([df_term, df_row])

        area.extend(list(loop(df_term, name='T', show=show)))
    return area

def process(input : dict):
    data_edf = input["data_edf"]
    n_term_start = input["n_term_start"]
    n_term_finish = input["n_term_finish"] 
    filt = input["filt"]
    f_sreza = input["f_sreza"]
    Fs_new = input["f_sampling"]
    show_detect_pqrst = input["show_detected_pqrst"]
    show_ECG = input["show_ecg"]
    plot_3D = input["plot_3d"]
    save_images = input["save_images"]
    show_log_scaling = input["show_log_scaling"]
    cancel_showing = input["cancel_showing"]
    QRS_loop_area = input["qrs_loop_area"]
    T_loop_area = input["t_loop_area"]
    show_log_loop_area = input["show_log_loop_area"]

    if cancel_showing:
        show_detect_pqrst = False
        show_ECG = False
        plot_3D = False
        show_log_scaling = False
        show_log_loop_area = False

    if n_term_finish != None:
        if n_term_finish < n_term_start:
            raise ValueError("Ошибка: n_term_finish должно быть >= n_term_start")
        else:
          n_term = [n_term_start, n_term_finish]  
    else:
        n_term = n_term_start

    # Считывание edf данных:
    data = mne.io.read_raw_edf(data_edf, verbose=0)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    fd = info['sfreq'] # Частота дискретизации
    df = pd.DataFrame(data=raw_data.T,    # values
                index=range(raw_data.shape[1]),  # 1st column as index
                columns=channels)  # 1st row as the column names
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
        channels = df.columns
        
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    # Ресемлинг:
    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        new_ecg, time_new = discrete_signal_resample(sig, t, Fs_new)
        df_new[graph] = pd.Series(new_ecg) 
    df = df_new.copy()

    # ФВЧ фильтрация артефактов дыхания:
    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, f_sreza, 'hp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()

    # Поиск точек PQRST:
    signal = np.array(df['ECG I'])
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=Fs_new, method="peak")

    # Отображение PQST точек на сигнале первого отведения
    if show_detect_pqrst:
        plt.figure(figsize=(12, 5))

        # Отобразим сигнал на графике
        plt.plot(time_new, signal, label='Signal', color='black')

        # Отобразим вертикальные линии для каждого типа точек
        for wave_type, peaks in waves_peak.items():
            if wave_type in ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']:
                wave_type_label = wave_type.split('_')[1]  # Извлекаем часть имени для метки графика
                for peak in peaks:
                    if not np.isnan(peak):  # Проверяем, что значение точки не является NaN
                        if wave_type == 'ECG_P_Peaks':
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted', color='red', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_Q_Peaks':
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted', color='green', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_S_Peaks': 
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted', color='m', label=f'{wave_type_label} Peak')
                        else:  
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted', color='blue', label=f'{wave_type_label} Peak')

        plt.xlim([0, 6])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal ECG I')
        plt.title('Детекция PQRST на 1 отведении')
        plt.show()

    # Отображение многоканального ЭКГ с детекцией R зубцов
    if show_ECG:
        num_channels = len(channels)
        fig, axs = plt.subplots(int(num_channels/2), 2, figsize=(11, 8), sharex=True)

        for i, graph in enumerate(channels):
            row = i // 2
            col = i % 2

            sig = np.array(df[graph])

            axs[row, col].plot(time_new, sig)
            axs[row, col].scatter(time_new[rpeaks['ECG_R_Peaks']], sig[rpeaks['ECG_R_Peaks']], color='red')
            axs[row, col].set_title(graph)
            axs[row, col].set_xlim([0, 6])
            axs[row, col].set_title(graph)
            axs[row, col].set_xlabel('Time (seconds)')

        plt.tight_layout()
        plt.show()

    # ВЫбор исследуемого периода/периодов
    i = n_term
    if type(i) == list:
        print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]
    else:
        print(f"Запрошен {i} период")
        fin = i
        beg = i

    start = rpeaks['ECG_R_Peaks'][beg-1]
    end = rpeaks['ECG_R_Peaks'][fin]
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    df_term = pd.concat([df_term, df_row])
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']

    # Расчет ВЭКГ
    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    df_term['size'] = 100

    # Построение проекций ВЭКГ:
    if not cancel_showing:
        plt.figure(figsize=(15, 5), dpi=100)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.x,df_term.y)
        plt.title('Фронтальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 3, 2)
        plt.plot(df_term.y,df_term.z)
        plt.title('Сагитальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.x, df_term.z)
        plt.title('Аксиальная плоскость')  
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.show()

    # Интерактивное 3D отображение
    if plot_3D:
        fig = px.scatter_3d(df_term, x='x', y='y', z='z', size='size', size_max=10, opacity=1)
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
    area = None
    # Работа при указании одного периода ЭКГ: 
    if  n_term_finish == None or n_term_finish == n_term_start:
        ## Масштабирование:
        # Поиск центра масс:
        x_center = df_term.x.mean()
        y_center = df_term.y.mean()
        z_center = df_term.z.mean()

        df_term['x_scaled'] = df_term.x - x_center
        df_term['y_scaled'] = df_term.y - y_center
        df_term['z_scaled'] = df_term.z - z_center

        # Нормирование на максимальное значение 
        max_value = max(df_term['x_scaled'].abs().max(), df_term['y_scaled'].abs().max(), df_term['z_scaled'].abs().max())
        df_term['x_scaled'] = df_term['x_scaled'] / max_value
        df_term['y_scaled'] = df_term['y_scaled'] / max_value
        df_term['z_scaled'] = df_term['z_scaled'] / max_value

        # Показ логов масштабирования
        if show_log_scaling:
            plt.figure(figsize=(8, 10), dpi=80)
            plt.subplot(3, 2, 1)
            plt.plot(df_term.x, df_term.y)
            plt.title('Исходные проекции')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.plot(x_center, y_center, marker='*', markersize=11, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 2)
            plt.plot(df_term.x_scaled, df_term.y_scaled)
            plt.title('Масштабированные проекции')
            plt.xlabel('X')
            plt.ylabel('Y') 
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.grid(True)

            plt.subplot(3, 2, 3)
            plt.plot(df_term.y, df_term.z)
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.plot(y_center, z_center, marker='*', markersize=11, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 4)
            plt.plot(df_term.y_scaled, df_term.z_scaled)
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.xlabel('Y')
            plt.ylabel('Z')
            plt.grid(True)

            plt.subplot(3, 2, 5)
            plt.plot(df_term.x, df_term.z)
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.plot(x_center, z_center, marker='*', markersize=12, label='Центр масс', color='red')
            plt.grid(True)
            plt.legend()

            plt.subplot(3, 2, 6)
            plt.plot(df_term.x_scaled, df_term.z_scaled)
            plt.xlabel('X')
            plt.ylabel('Z')
            plt.xlim([-1.05, 1.05])
            plt.ylim([-1.05, 1.05])
            plt.grid(True)
            plt.show()

        # Поиск площадей при задании на исследование одного периодка ЭКГ:
        area = get_area(show=show_log_loop_area, df=df, waves_peak=waves_peak,
                         start=start, Fs_new=Fs_new,  QRS=QRS_loop_area, T=T_loop_area)
        

    # Сохранение масштабированных изображений
    if save_images and (n_term_finish == None or n_term_finish == n_term_start):
        file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
        name = f'{file_name_without_extension}_period_{n_term_start}.png'
        
        # Создадим папки для записи если их еще нет:
        if not os.path.exists('saved_vECG'):
            os.makedirs('saved_vECG')
        if not os.path.exists('saved_vECG/frontal_plane'):
            os.makedirs('saved_vECG/frontal_plane')
        if not os.path.exists('saved_vECG/sagittal_plane'):
            os.makedirs('saved_vECG/sagittal_plane')
        if not os.path.exists('saved_vECG/axial_plane'):
            os.makedirs('saved_vECG/axial_plane')      

        # После каждого plt.show() добавим код для сохранения графика в ЧБ формате
        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.x_scaled, df_term.y_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/frontal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.y_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/sagittal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])  
        plt.plot(df_term.x_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/axial_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()
        print('Фотографии сохранены в папке saved_vECG')

    return area



if __name__ == "__main__":
    process()