import mne
import math
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
import plotly.graph_objects as go
import click
import torch
from PIL import Image
from torchvision import transforms
from models_for_inference.model import *
import warnings


def convert_to_posix_path(windows_path):
    # Перевод пути к формату posix:
    posix_path = windows_path.replace('\\', '/')
    return posix_path


def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def discrete_signal_resample_for_DL(signal, old_sampling_rate, new_sampling_rate):
    ## Производит ресемплирование для DL преподготовки
    # Вычисляем коэффициент, определяющий отношение новой частоты к старой
    resample_factor = new_sampling_rate / old_sampling_rate

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * resample_factor)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)

    return new_signal


def discrete_signal_resample(signal, time, new_sampling_rate):
    ## Производит ресемплирование
    # Текущая частота дискретизации
    current_sampling_rate = 1 / np.mean(np.diff(time))

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * new_sampling_rate / current_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)
    new_time = np.linspace(time[0], time[-1], num_points_new)

    return new_signal, new_time


def calculate_area(points):
    # Считает площадь замкнутого полигона
    polygon = Polygon(points)
    area_inside_loop = polygon.area
    return area_inside_loop


def find_mean(df_term):
    # Считает средние значения петель
    x_center = df_term.x.mean()
    y_center = df_term.y.mean()
    z_center = df_term.z.mean()
    return [x_center, y_center, z_center]


def find_qrst_angle(mean_qrs, mean_t, name=''):
    ## Находит угол QRST с помощью скалярного произведения
    # Преобразуем списки в numpy массивы
    mean_qrs = np.array(mean_qrs)
    mean_t = np.array(mean_t)

    # Находим угол между векторами в радианах
    dot_product = np.dot(mean_qrs, mean_t)
    norm_qrs = np.linalg.norm(mean_qrs)
    norm_t = np.linalg.norm(mean_t)
    angle_radians = np.arccos(dot_product / (norm_qrs * norm_t))

    # Конвертируем угол из радиан в градусы
    angle_degrees = np.degrees(angle_radians)
    print(f"Угол QRST {name}равен {round(angle_degrees, 2)} градусов")

    return angle_degrees


def make_vecg(df_term):
    # Получает значения ВЭКГ из ЭКГ
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
    return df_term

    
def loop(df_term, name, show=False):
    # Подсчет и отображение площади петли
    if name == 'T':
        name_loop = 'ST-T'
    else:
        name_loop = name

    if show:
        plt.figure(figsize=(29, 7), dpi=68)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.y, df_term.z)
        plt.title('Фронтальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 2)
        plt.gca().invert_xaxis()
        plt.plot(df_term.x, df_term.z)
        plt.title('Сагиттальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.y, df_term.x)
        plt.title('Аксиальная плоскость')  
        plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('X')

        plt.suptitle(f'{name_loop} петля', fontsize=16)
        plt.show()
    
    points = list(zip(df_term['x'], df_term['y']))
    area_inside_loop_1 = calculate_area(points)
    print(f"Площадь петли {name_loop} во фронтальной плоскости:", area_inside_loop_1)

    points = list(zip(df_term['y'], df_term['z']))
    area_inside_loop_2 = calculate_area(points)
    print(f"Площадь петли {name_loop} в сагиттальной плоскости:", area_inside_loop_2)

    points = list(zip(df_term['x'], df_term['z']))
    area_inside_loop_3 = calculate_area(points)
    print(f"Площадь петли {name_loop} в аксиальной плоскости:", area_inside_loop_3)

    return area_inside_loop_1, area_inside_loop_2, area_inside_loop_3


def get_area(show, df, waves_peak, start, Fs_new, QRS, T):
    # Выделяет области петель для дальнейшей обработки - подсчета угла QRST и площадей
    area = []
    # Уберем nan:
    waves_peak['ECG_Q_Peaks'] = [x for x in waves_peak['ECG_Q_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_S_Peaks'] = [x for x in waves_peak['ECG_S_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_T_Offsets'] = [x for x in waves_peak['ECG_T_Offsets'] if not math.isnan(x)]   

    # QRS петля
    # Ищем ближний пик к R пику
    closest_Q_peak = min(waves_peak['ECG_Q_Peaks'], key=lambda x: abs(x - start))
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
    df_new = df.copy()
    df_term = df_new.iloc[closest_Q_peak:closest_S_peak,:]
    df_row = df_new.iloc[closest_Q_peak:closest_Q_peak+1,:]
    df_term = pd.concat([df_term, df_row])
    #df_term = make_vecg(df_term)
    mean_qrs = find_mean(df_term)
    if QRS:
        area = list(loop(df_term, name='QRS', show=show))

    ## ST-T петля
    # Ищем ближний пик к R пику
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
    # Ищем ближний пик к S пику
    closest_T_end = min(waves_peak['ECG_T_Offsets'], key=lambda x: abs(x - closest_S_peak))
    df_new = df.copy()
    df_term = df_new.iloc[closest_S_peak + int(0.025*Fs_new) : closest_T_end, :]
    df_row = df_new.iloc[closest_S_peak+int(0.025*Fs_new):closest_S_peak+int(0.025*Fs_new)+1,:]
    df_term = pd.concat([df_term, df_row])
    #df_term = make_vecg(df_term)
    mean_t = find_mean(df_term)
    if T:
        area.extend(list(loop(df_term, name='T', show=show)))
    return area, mean_qrs, mean_t


def preprocessing_3d(list_coord):
    # Строит линии на 3D графике, отвечающие за вектора средних ЭДС петель
    A = np.array(list_coord)

    step = 0.025
    # Создаем массив точек от (0, 0, 0) до точки A с заданным шагом
    interpolated_points = []
    for t in np.arange(0, 1, step):
        interpolated_point = t * A
        interpolated_points.append(interpolated_point)

    # Добавляем точку A в конец массива
    interpolated_points.append(A)

    # Преобразуем список точек в numpy массив
    interpolated_points = np.array(interpolated_points)

    df = pd.DataFrame(interpolated_points, columns=['x', 'y', 'z'])
    df['s']=20 # задали размер для 3D отображения
    return df


def angle_3d_plot(df1, df2, df3):
    # Построение интерактивного графика логов вычисления угла QRST 
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df1['x'],
            y=df1['y'],
            z=df1['z'],
            mode='markers',
            marker=dict(size=df1['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила QRS'
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df2['x'],
            y=df2['y'],
            z=df2['z'],
            mode='markers',
            marker=dict(size=df2['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила ST-T'
        )
    )
    df3['size'] = 10
    fig.add_trace(
        go.Scatter3d(
            x=df3['x'],
            y=df3['y'],
            z=df3['z'],
            mode='markers',
            marker=dict(size=df3['size'], sizemode='diameter', opacity=1),
            name='ВЭКГ'
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def apply_filter_mean(column, window_size):
    # Фильтр среднего для сглаживания петли ST-T
    filtered_column = []

    for i in range(len(column)):
        if i < window_size // 2 or i >= len(column) - window_size // 2:
            filtered_column.append(column[i])
        else:
            window = column[i - window_size // 2:i + window_size // 2 + 1]
            filtered_value = np.mean(window)
            filtered_column.append(filtered_value)

    return filtered_column





#------------------------------------------ГЛАВНЫЙ КОД--------------------------------------#

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
    help="""Параметр задается исключительно при необходимости построить диапазон
      периодов. Значение является номером периода, до которого 
      будет вестись запись вЭКГ (включительно)""",
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
    help="""Включение/выключение режима для построения ключевых точек PQRST
      для сигнала ЭКГ, полученных с помощью дискретных вейвлет
      преобразований. По умолчанию режим отключен""",
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
    "--qrs_loop_area",
    help="""Включение/выключение режима для расчета площади QRS петли по всем проекциям.
    Работает при отображении лишь одного периода ЭКГ. По умолчанию режим включен""",
    default=True,
    type=bool,
)
@click.option(
    "--t_loop_area",
    help="""Включение/выключение режима для расчета площади ST-T петли по всем проекциям. 
    Работает при отображении лишь одного периода ЭКГ.
    (PS: Рассчет является менее точным, чем QRS петли из-за множественных 
    самопересечений) По умолчанию режим выключен""",
    default=False,
    type=bool,
)
@click.option(
    "--show_log_loop_area",
    help="""Включение/выключение режима для отображения отдельных петель. Доступен при
    включенной опции расчета площади какой-либо петли QRS_loop_area или T_loop_area
    По умолчанию режим выключен""",
    default=False,
    type=bool,
)
@click.option(
    "--count_qrst_angle",
    help="""Включение/выключение режима для вычисления пространственного угла QRST,
      а также проекции угла на фронтальную плоскость. Работает 
      при отображении лишь одного периода ЭКГ. По умолчанию режим включен""",
    default=True,
    type=bool,
)
@click.option(
    "--show_log_qrst_angle",
    help="""Включение/выключение режима для трехмерного отображения угла QRST на ВЭКГ.
      Работает при count_qrst_angle=True. По умолчанию режим выключен""",
    default=False,
    type=bool,
)
@click.option(
    "--save_images",
    help="""Включение/выключение режима для сохранения графиков вЭКГ трех
      плоскостей в качестве png изображений. Сохранение производится в папку saved_vECG,
      создающуюся в корне репозитория. Работает при отображении лишь одного 
      периода ЭКГ. По умолчанию режим отключен""",
    default=False,
    type=bool,
)
@click.option(
    "--show_log_scaling",
    help="""Включение/выключение режима для демонстрации логов масштабирования
      ВЭКГ для сохранения их как изображений с исходными пропорциями. Работает 
      при отображении лишь одного периода ЭКГ. По умолчанию режим отключен.""",
    default=False,
    type=bool,
)
@click.option(
    "--cancel_showing",
    help="""Включение/выключение режима для вывода любых графиков. Позволяет
      выключить отображение графических результатов для возможности
      использовать get_VECG в цикле по файлам ЭКГ. По умолчанию режим 
      отключен (то есть отображение графиков включено).""",
    default=False,
    type=bool,
)
@click.option(
    "--mean_filter",
    help="""Включение/выключение фильтра среднего для ST-T петли чтобы сгладить
    По умолчанию режим включен.""",
    default=True,
    type=bool,
)
@click.option(
    "--predict",
    help="""Включение/выключение СППР на основе PointNet и Resnet""",
    default=True,
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
    show_log_scaling = kwargs["show_log_scaling"]
    cancel_showing = kwargs["cancel_showing"]
    QRS_loop_area = kwargs["qrs_loop_area"]
    T_loop_area = kwargs["t_loop_area"]
    show_log_loop_area = kwargs["show_log_loop_area"]
    count_qrst_angle = kwargs["count_qrst_angle"]
    show_log_qrst_angle = kwargs["show_log_qrst_angle"]
    mean_filter = kwargs["mean_filter"]
    predict_res = kwargs["predict"]

    ## СЛЕДУЕТ УБРАТЬ ПРИ ТЕСТИРОВАНИИ:
    # Устанавливаем фильтр для игнорирования всех RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Включаем режим, позволяющий открывать графики сразу все
    plt.ion()

    if cancel_showing:
        show_detect_pqrst = False
        show_ECG = False
        plot_3D = False
        show_log_scaling = False
        show_log_loop_area = False
        show_log_qrst_angle = False

    if n_term_finish != None:
        if n_term_finish < n_term_start:
            raise ValueError("Ошибка: n_term_finish должно быть >= n_term_start")
        else:
          n_term = [n_term_start, n_term_finish]  
    else:
        n_term = n_term_start

    if '\\' in data_edf:
        # Преобразуем путь в формат Posix
        data_edf = convert_to_posix_path(data_edf)

    # Считывание edf данных:
    data = mne.io.read_raw_edf(data_edf, verbose=0)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names
    fd = info['sfreq'] # Частота дискретизации
    df = pd.DataFrame(data=raw_data.T,    
                index=range(raw_data.shape[1]),  
                columns=channels) 
    # Переименование столбцов при необходимости:
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
        channels = df.columns

    # Создание массива времени    
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
        
    # ФНЧ фильтрация (по желанию можно включить):
    filt_low_pass = False
    if filt_low_pass:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, 100, 'lp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()

    ## Поиск точек PQRST:
    n_otvedenie = 'I'
    signal = np.array(df['ECG I'])  

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)

    # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size <= 5:
        print("На I отведении не удалось детектировать R зубцы")
        print("Проводим детектирование по II отведению:")
        n_otvedenie = 'II'
        signal = np.array(df['ECG II'])  
        signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)
        
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size <= 3:
            print('Сигналы ЭКГ слишком шумные для анализа')
            # Отобразим эти шумные сигналы:
            if not cancel_showing:
                num_channels = len(channels)
                fig, axs = plt.subplots(int(num_channels/2), 2, figsize=(11, 8), sharex=True)
                for i, graph in enumerate(channels):
                    row = i // 2
                    col = i % 2
                    sig = np.array(df[graph])
                    axs[row, col].plot(time_new, sig)
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlim([0, 6])
                    axs[row, col].set_title(graph)
                    axs[row, col].set_xlabel('Time (seconds)')
                plt.tight_layout()
                plt.show()
                plt.ioff()
                plt.show()
            return # Выход из функции досрочно

    # Поиск точек pqst:
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=Fs_new, method="peak")

    # Отображение PQST точек на сигнале первого отведения (или второго при ошибке на первом)
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
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='red', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_Q_Peaks':
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='green', label=f'{wave_type_label} Peak')
                        elif wave_type == 'ECG_S_Peaks': 
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='m', label=f'{wave_type_label} Peak')
                        else:  
                            plt.axvline(x=time_new[int(peak)], linestyle='dotted',
                                        color='blue', label=f'{wave_type_label} Peak')
        plt.xlim([0.5, 6])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Signal ECG I')
        plt.title(f'Детекция PQRST на {n_otvedenie} отведении')
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
            axs[row, col].scatter(time_new[rpeaks['ECG_R_Peaks']], 
                                  sig[rpeaks['ECG_R_Peaks']], color='red')
            axs[row, col].set_title(graph)
            axs[row, col].set_xlim([0, 6])
            axs[row, col].set_title(graph)
            axs[row, col].set_xlabel('Time (seconds)')

        plt.tight_layout()
        plt.show()

    # Выбор исследуемого периода/периодов
    i = n_term
    if type(i) == list:
        print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]
    else:
        print(f"Запрошен {i} период")
        fin = i
        beg = i

    if beg-1 < 0 or fin >= len(rpeaks['ECG_R_Peaks']):
        print('Запрашиваемого перода/диапазона периодов не существует')
        return # Выход из функции досрочно
    
    start = rpeaks['ECG_R_Peaks'][beg-1]
    end = rpeaks['ECG_R_Peaks'][fin]
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    df_term = pd.concat([df_term, df_row])

    # Расчет ВЭКГ
    df_term = make_vecg(df_term)
    df = make_vecg(df)
    df_term['size'] = 100 # задание размера для 3D визуализации

    if mean_filter:
        df = make_vecg(df)
        window = int(Fs_new * 0.02)
        df['x'] = apply_filter_mean(np.array(df['x']), window)
        df['y'] = apply_filter_mean(np.array(df['y']), window)
        df['z'] = apply_filter_mean(np.array(df['z']), window)
        df_term = df.iloc[start:end,:]
        df_row = df.iloc[start:start+1,:]
        df_term = pd.concat([df_term, df_row])
        df_term['size'] = 100 
        
    # Построение проекций ВЭКГ:
    if not cancel_showing:
        plt.figure(figsize=(29, 7), dpi=68)
        plt.subplot(1, 3, 1)
        plt.plot(df_term.y, df_term.z)
        plt.title('Фронтальная плоскость')
        plt.xlabel('Y')
        plt.ylabel('Z')

        plt.subplot(1, 3, 2)
        plt.gca().invert_xaxis()
        plt.plot(df_term.x, df_term.z)
        plt.title('Сагиттальная плоскость')
        plt.xlabel('X')
        plt.ylabel('Z')

        plt.subplot(1, 3, 3)
        plt.plot(df_term.y, df_term.x)
        plt.title('Аксиальная плоскость')  
        plt.gca().invert_yaxis()
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.show()

    # Интерактивное 3D отображение
    if plot_3D:
        fig = px.scatter_3d(df_term, x='x', y='y', z='z', size='size', size_max=10, opacity=1)
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

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
        max_value = max(df_term['x_scaled'].abs().max(),
                        df_term['y_scaled'].abs().max(),
                        df_term['z_scaled'].abs().max())
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

        # СППР:
        # Инференс модели pointnet:
        if predict_res:
            point_cloud_array_innitial = df_term[['x', 'y', 'z']].values
            
            # Приведем к дискретизации 700 Гц на котором обучалась сеть
            new_num_points = int(len(point_cloud_array_innitial) * 700 / Fs_new)

            # Инициализируем новый массив
            point_cloud_array = np.zeros((new_num_points, 3))

            # Производим ресемплирование каждой координаты
            for i in range(3):
                point_cloud_array[:, i] = discrete_signal_resample_for_DL(point_cloud_array_innitial[:, i], Fs_new, 700)

            # Трансформация входных данных
            val_transforms = transforms.Compose([
                        Normalize(),
                        PointSampler_weighted(512),
                        ToTensor()
                        ])
            inputs = val_transforms(point_cloud_array)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.double()

            pointnet = PointNet().double()
            # Загрузка сохраненных весов модели
            pointnet.load_state_dict(torch.load('models_for_inference/pointnet.pth'))
            pointnet.eval().to('cpu')
            # инференс:
            with torch.no_grad():
                outputs, __, __ = pointnet(inputs.transpose(1,2))

                softmax_outputs = torch.softmax(outputs, dim=1)
                probabilities, predicted_class = torch.max(softmax_outputs, 1)

            if predicted_class == 0:
                message = f'Здоров (уверенность PointNet {probabilities.item() * 100:.2f}%)'
            else:
                message = f'Болен (уверенность PointNet {probabilities.item() * 100:.2f}%)'
            print(message)


            # Инференс ResNet
            file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
            name = '.png'
            
            # После каждого plt.show() добавим код для сохранения графика в ЧБ формате
            plt.figure(figsize=(7, 7), dpi=150)
            plt.xlim([-1.03, 1.03])
            plt.ylim([-1.03, 1.03])
            plt.plot(df_term.x_scaled, df_term.y_scaled, color='black')
            plt.axis('off')  # Отключить оси и подписи
            name_save = 'XY_plane' + name
            plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
            plt.close()

            plt.figure(figsize=(7, 7), dpi=150)
            plt.xlim([-1.03, 1.03])
            plt.ylim([-1.03, 1.03])
            plt.plot(df_term.y_scaled, df_term.z_scaled, color='black')
            plt.axis('off')  # Отключить оси и подписи
            name_save = 'YZ_plane' + name
            plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
            plt.close()

            plt.figure(figsize=(7, 7), dpi=150)
            plt.xlim([-1.03, 1.03])
            plt.ylim([-1.03, 1.03])  
            plt.plot(df_term.x_scaled, df_term.z_scaled, color='black')
            plt.axis('off')  # Отключить оси и подписи
            name_save = 'XZ_plane' + name
            plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
            plt.close()
            
            r_image = Image.open('XZ_plane' + name).convert("L")
            g_image = Image.open('XY_plane' + name).convert("L")
            b_image = Image.open('YZ_plane' + name).convert("L")

            # Создайте цветное изображение RGB
            color_image = Image.merge('RGB', (r_image, g_image, b_image))

            # Сохраните цветное изображение в новой папке
            color_image.save('combined.png')

            # Загрузка сохраненной модели
            model = torch.jit.load("models_for_inference/resnet.pt").to('cpu').eval()

            # Задаем преобразования: изменение размера, нормализация и преобразование в тензор
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            # Загрузка изображения
            image = Image.open('combined.png').convert("RGB")

            # Применение преобразований
            input_tensor = transform(image).unsqueeze(0).to('cpu')  # Добавляем размерность пакета

            # Пропуск изображения через модель
            with torch.no_grad():
                outputs = model(input_tensor)
                softmax_outputs = torch.softmax(outputs, dim=1)
                probabilities, predicted_class = torch.max(softmax_outputs, 1)

            if predicted_class == 0:
                message = f'Здоров (уверенность ResNet {probabilities.item() * 100:.2f}%)'
            else:
                message = f'Болен (уверенность ResNet {probabilities.item() * 100:.2f}%)'
            print(message)
            os.remove('combined.png')
            os.remove('XY_plane.png')
            os.remove('XZ_plane.png')
            os.remove('YZ_plane.png')



        # Поиск площадей при задании на исследование одного периодка ЭКГ:
        area_projections , mean_qrs, mean_t = get_area(show=show_log_loop_area, df=df,
                                                       waves_peak=waves_peak, start=start,
                                                       Fs_new=Fs_new,  QRS=QRS_loop_area, 
                                                       T=T_loop_area)
        # Определение угла QRST:
        if count_qrst_angle:
            angle_qrst = find_qrst_angle(mean_qrs, mean_t)
            angle_qrst_front = find_qrst_angle(mean_qrs[:2], mean_t[:2],
                                               name='во фронтальной плоскости ')

            # Отображение трехмерного угла QRST
            if show_log_qrst_angle:
                df_qrs = preprocessing_3d(mean_qrs)
                df_t = preprocessing_3d(mean_t)
                angle_3d_plot(df_qrs, df_t, df_term)

    # Сохранение масштабированных изображений
    if save_images and (n_term_finish == None or n_term_finish == n_term_start):
        file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
        name = f'{file_name_without_extension}_period_{n_term_start}.png'
        
        # Создадим папки для записи если их еще нет:
        if not os.path.exists('saved_vECG'):
            os.makedirs('saved_vECG')
        if not os.path.exists('saved_vECG/XY_plane'):
            os.makedirs('saved_vECG/XY_plane')
        if not os.path.exists('saved_vECG/YZ_plane'):
            os.makedirs('saved_vECG/YZ_plane')
        if not os.path.exists('saved_vECG/XZ_plane'):
            os.makedirs('saved_vECG/XZ_plane')      

        # После каждого plt.show() добавим код для сохранения графика в ЧБ формате
        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.x_scaled, df_term.y_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/XY_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])
        plt.plot(df_term.y_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/YZ_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()

        plt.figure(figsize=(7, 7), dpi=150)
        plt.xlim([-1.03, 1.03])
        plt.ylim([-1.03, 1.03])  
        plt.plot(df_term.x_scaled, df_term.z_scaled, color='black')
        plt.axis('off')  # Отключить оси и подписи
        name_save = 'saved_vECG/XZ_plane/' + name
        plt.savefig(name_save, bbox_inches='tight', pad_inches=0, transparent=True, facecolor='white')
        plt.close()
        print('Фотографии сохранены в папке saved_vECG')

    # Выключаем интерактивный режим, чтобы окна графиков не закрывались сразу
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()