# Проект по векторной электрокардиографии

Векторная электрокардиография (вЭКГ) - это метод, позволяющий измерять и представлять электрический вектор сердца во время сердечного цикла. Этот вектор представляет собой направление дипольного момента сердца, что дает информацию о сокращении сердечной мышцы. Врачи используют ВЭКГ для анализа движения вектора в трех основных плоскостях и его 3D отображения для диагностики и мониторинга состояния сердца. Такая информация может быть полезной для обнаружения аномалий, нарушений проводимости и оценки эффективности лечения. Векторная электрокардиография является важным инструментом в кардиологии и помогает улучшить диагностику и лечение сердечно-сосудистых заболеваний.

В данном проекте представлена программа, позволяющая производить построение вЭКГ в трех плоскостях, а также дающая возможность интерактивно взаимодействовать с 3D отображением. Для более гибкой настройки имеется возможность включать/выключать фильтрацию артефактов дыхания, менять частоту дискретизации исходного сигнала, выбирать интересующие периоды кардиоцикла и находить основные точки PQRST на сигналах всех отведений. Так же добавлена возможность производить сохранение полученных проекций вЭКГ на главные плоскости в виде масштабированных grayscale изображений для создания датасета для дальнейших исследований и ML разработок. Данная программа позволяет проводить вычисление площадей проекций петель QRS и ST-T, а также вычислять пространственный и фронтальный угол QRS-T.

Подробный pdf отчет доступен по ссылке - [__ОТЧЕТ__](https://github.com/Koldim2001/vector_ECG/blob/main/utils/Отчет%20по%20проекту%20get_VECG.pdf) <br/>
Видео туториал доступен по ссылке - [__get_VECG туториал__](https://youtu.be/t_-8pExz1gs)<br/>
Презентация новых возможностей проекта - [__get_VECG_2.0__](https://youtu.be/8WBRQ-31ZdI)

## Installation:
```
git clone https://github.com/Koldim2001/vector_ECG.git
```
```
cd vector_ECG
```
```
pip install -e .
```

## Как запускать код:

__Классический подход c предустановленными параметрами:__
```
get_VECG --data_edf="Data_VECG\ECG_1.edf" 
```
__Более тонкая настройка параметров:__

Задание диапазона периодов ЭКГ для обработки:
```
get_VECG --data_edf="Data_VECG\ECG_1.edf" --n_term_start=2 --n_term_finish=6 --filt=True --f_sreza=0.5 --f_sampling=2000 --show_detected_pqrst=True --show_ecg=True --plot_3d=True --mean_filter=True
```

Задание одного периода ЭКГ для обработки и сохранения результатов ВЭКГ в виде png файлов масштабированных проекций:
```
get_VECG --data_edf="Data_VECG\ECG_1.edf" --n_term_start=3 --filt=True --f_sreza=0.5 --f_sampling=2500 --show_detected_pqrst=False --show_ecg=False --plot_3d=False --qrs_loop_area=False --t_loop_area=False --show_log_loop_area=False --count_qrst_angle=False --show_log_qrst_angle=False --save_images=True --show_log_scaling=True --mean_filter=False --predict=False
```
Задание одного периода ЭКГ для обработки c выводом всех логов и найденных углов с площадями:
```
get_VECG --data_edf="Data_VECG\ECG_1.edf" --n_term_start=3 --filt=True --f_sreza=0.5 --f_sampling=2500 --show_detected_pqrst=True --show_ecg=True --plot_3d=True --qrs_loop_area=True --t_loop_area=True --show_log_loop_area=True --save_images=False --show_log_scaling=True --count_qrst_angle=True --show_log_qrst_angle=True --mean_filter=False --predict=True
```

Список параметров с пояснениями, которые можно передать на вход программы перед ее запуском в cli:
```bash
--data_edf TEXT                Путь к файлу ЭКГ формата .edf

--n_term_start INTEGER         Номер интересующего для анализа периода кардиоцикла. Если необходимо
                               построить результат за диапазон периодов, то данный параметр задает 
                               стартовое значение номера периода для записи в ЭКГ

--n_term_finish INTEGER        Параметр задается исключительно при необходимости построить диапазон 
                               периодов. Значение является номером периода, до которого будет
                               вестись запись в ЭКГ (включительно)

--filt BOOL                    Включение/выключение цифровой фильтрации исходных сигналов с помощью
                               фильтра Баттерворта 1 порядка (ФВЧ). По умолчанию фильтрация отключена

--f_sreza FLOAT                Задание частоты среза ФВЧ фильтра. Используется исключительно при 
                               выборе режима --filt=True. По умолчанию = 0.7 Гц

--f_sampling FLOAT             Задание частоты дискретизации. Будет проведено ресемплирование исходного
                               сигнала с использованием линейной интерполяции. По умолчанию Fs=1500 Гц

--show_detected_pqrst BOOL     Включение/выключение режима для построения ключевых точек PQRST для
                               сигнала ЭКГ, полученных с помощью дискретных вейвлет преобразований.
                               По умолчанию режим отключен

--show_ecg BOOL                Включение/выключение режима для построения графиков всех отведений и
                               обнаруженных QRS пиков, относительно которых ведется подсчет номеров
                               n_term_start и n_term_finish. По умолчанию режим отключен

--plot_3d BOOL                 Включение/выключение режима для интерактивного отображения 3D графика
                               в ЭКГ. По умолчанию режим включен

--qrs_loop_area BOOL           Включение/выключение режима для расчета площади QRS петли по 
                               всем проекциям.  Работает при отображении лишь одного периода ЭКГ.
                               По умолчанию режим включен                         

--t_loop_area BOOL             Включение/выключение режима для расчета площади ST-T петли по всем 
                               проекциям. Работает при отображении лишь одного периода ЭКГ. 
                               (PS: Рассчет является менее точным, чем QRS петли из-за 
                               множественных самопересечений) По  умолчанию режим отключен
                                
--show_log_loop_area BOOL      Включение/выключение режима для отображения отдельных петель.
                               Доступен при включенной опции расчета площади какой-либо петли
                               QRS_loop_area или T_loop_area. По умолчанию режим отключен

--count_qrst_angle BOOL        Включение/выключение режима для вычисления пространственного угла
                               QRST, а также проекции угла на фронтальную плоскость. Работает при 
                               отображении лишь одного периода ЭКГ. По умолчанию режим включен
                               
--show_log_qrst_angle BOOL     Включение/выключение режима для трехмерного отображения угла QRST на
                               ВЭКГ. Работает при count_qrst_angle=True. По умолчанию режим отключен
            
--save_images BOOL             Включение/выключение режима для сохранения графиков в ЭКГ трех плоскостей
                               в качестве png изображений. Сохранение производится в папку saved_vECG,
                               создающуюся в корне репозитория. Работает при отображении лишь одного
                               периода ЭКГ. По умолчанию режим отключен

--show_log_scaling BOOL        Включение/выключение режима для демонстрации логов масштабирования
                               ВЭКГ для сохранения их как изображений с исходными пропорциями. Работает 
                               при отображении лишь одного периода ЭКГ. По умолчанию режим отключен

--cancel_showing BOOL          Включение/выключение режима для вывода любых графиков. Позволяет
                               выключить отображение графических результатов для возможности
                               использовать get_VECG в цикле по файлам ЭКГ. По умолчанию режим 
                               отключен (то есть отображение графиков включено)

--mean_filter BOOL             Включение/выключение фильтра среднего для ST-T петли 
                               чтобы сгладить По умолчанию режим включен.

--predict BOOL                 Включение/выключение СППР на основе PointNet и ResNet

--help                         Покажет существующие варианты парсинга аргументов в CLI

```
