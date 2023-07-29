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
import click


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



@click.command()
@click.option(
    "--data_edf",
    default="Data_VECG/ECG_1.edf",
    help="Путь к файлу ЭКГ формата .edf",
    type=str,
)
@click.option(
    "--n_term_start",
    default=3,
    help="""Номер интересующего для анализа периода кардиоцикла.
      Если необходимо построить результат за диапазон периодов, то данный 
      параметр задает стартовое значение номера периода для записи вЭКГ
      """,
    type=int,
)
@click.option(
    "--n_term_finish",
    default=None,
    help="""Параметр задается исключительно при необходимости построить диапазон периодов.
      Значение является номером периода, до которого будет вестись запись вЭКГ (включительно)""",
    type=int,
)
@click.option(
    "--filt",
    help="""Включение/выключение цифровой фильтрации исходных сигналов с помошью
      фильтра Баттерворта 1 порядка (ФВЧ). По умолчанию фильтрация отключена""",
    default=False,
    type=bool,
)
@click.option(
    "--f_sreza",
    help="""Задание частоты среза ФВЧ фильтра. Используется исключительно при
      выборе режима --filt=True. По умолчанию = 0.7 Гц""",
    default=0.7,
    type=float,
)

@click.option(
    "--f_sampling",
    help="""Задание частоты дискретизации. Будет проведено ресемплирование исходного
      сигнла с использованием линейной инетрполяции. По умолчанию Fs=1500 Гц""",
    default=1500.0,
    type=float,
)
@click.option(
    "--show_detected_pqrst",
    help="""Включение/выключение режима для построения ключевых точек PQRST для сигнала ЭКГ,
     полученных с помощью дискретных вейвлет преобразований. По умолчанию режим отключен""",
    default=False,
    type=bool,
)
@click.option(
    "--show_ecg",
    help="""Включение/выключение режима для построения графиков всех отведений и 
      обнаруженных QRS пиков, относительно которых ведется подсчет номеров
      n_term_start и n_term_finish. По умолчанию режим отключен""",
    default=False,
    type=bool,
)
@click.option(
    "--plot_3d",
    help="""Включение/выключение режима для интерактивного отображения 3D графика вЭКГ.
    По умолчанию режим включен""",
    default=True,
    type=bool,
)
@click.option(
    "--save_images",
    help="""Включение/выключение режима для сохранения графиков вЭКГ трех
      плоскостей в качестве png изображений. Сохранение производится в папку saved_vECG,
      создающуюся в корне репозитория. Работает при отображении лишь одного 
      периода кардиоцикла. По умолчанию режим По умолчанию режим отключен""",
    default=False,
    type=bool,
)
def main(**kwargs):
    # ------------------ ARG parse ------------------
    data_edf = kwargs["data_edf"]
    n_term_start = kwargs["n_term_start"]
    n_term_finish = kwargs["n_term_finish"] 
    filt = kwargs["filt"]
    f_sreza = kwargs["f_sreza"]
    Fs_new = kwargs["f_sampling"]
    show_detect_pqrst = kwargs["show_detected_pqrst"]
    show_ECG = kwargs["show_ecg"]
    plot_3D = kwargs["plot_3d"]
    save_images = kwargs["save_images"]


    if n_term_finish != None:
        if n_term_finish < n_term_start:
            raise ValueError("Ошибка: n_term_finish должно быть >= n_term_start")
        else:
          n_term = [n_term_start, n_term_finish]  
    else:
        n_term = n_term_start

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


    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        new_ecg, time_new = discrete_signal_resample(sig, t, Fs_new)
        df_new[graph] = pd.Series(new_ecg) 
    df = df_new.copy()

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

    # Extract R-peaks locations and PQRST
    signal = np.array(df['ECG I'])
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=Fs_new, method="peak")

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
        # Добавим подписи осей и заголовок графика
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal ECG I')
        plt.title('Детекция PQRST на 1 отведении')

        # Отобразим график
        plt.show()


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

        #plt.xlabel('Time (seconds)')
        plt.tight_layout()
        plt.show()


    # Подсчет вЭКГ
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

    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)

    df_term['size'] = 100

    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.x,df_term.y)
    plt.title('Фронтальная плоскость')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot()
    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.y,df_term.z)
    plt.title('Сагитальная плоскость')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.plot()
    plt.figure(figsize=(7, 7), dpi=80)
    plt.plot(df_term.x, df_term.z)
    plt.title('Аксиальная плоскость')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.show()

    if plot_3D:
        fig = px.scatter_3d(df_term, x='x', y='y', z='z', size='size', size_max=10, opacity=1)
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

    if save_images and n_term_finish == None:
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
        plt.figure(figsize=(7, 7), dpi=80)
        plt.plot(df_term.x, df_term.y, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/frontal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=80)
        plt.plot(df_term.y, df_term.z, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/sagittal_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=80)
        plt.plot(df_term.x, df_term.z, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/axial_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()
        print('Фотографии сохранены в папке saved_vECG')


if __name__ == "__main__":
    main()