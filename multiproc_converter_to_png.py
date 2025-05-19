import os
from multiprocessing import Pool, cpu_count
import cairosvg
from PIL import Image
import re
from io import BytesIO

def load_single_image(args):
    """
    Функция для загрузки одного изображения.

    :param args: кортеж (key, filepath)
    :return: (key, image_data) или (key, None) в случае ошибки
    """
    key, filepath = args

    if not os.path.exists(filepath):
        print(f"[Ошибка] Файл не найден: {filepath}")
        return key, None
    try:
        image = Image.open(filepath).convert("RGB")  # всегда RGB
        # Если хочешь вернуть сам объект Image, можно так:
        # return key, image
        # Или, например, numpy array:
        # return key, np.array(image)
        return key, image  # или сохранить как данные по желанию
    except Exception as e:
        print(f"[Ошибка при загрузке {filepath}]: {e}")
        return key, None

def load_images_parallel(image_dict, num_processes=None):
    """
    Параллельно загружает изображения из локальных путей.

    :param image_dict: dict {key: filepath}
    :param num_processes: int - кол-во процессов (по умолчанию: число ядер CPU)
    :return: dict {key: image_data}
    """
    items = list(image_dict.items())
    if not items:
        return {}
    if num_processes is None:
        num_processes = min(cpu_count(), len(items))
    #print(f"Загрузка {len(items)} изображений на {num_processes} процессах...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(load_single_image, items)
    # Преобразуем список (key, image) в словарь
    loaded_images = {key: image for key, image in results if image is not None}
    return loaded_images

def convert_svgs_to_pngs_parallel(svg_dict, output_folder, num_processes=None):
    """
    Основная функция. Конвертирует словарь SVG-строк в PNG файлы параллельно.

    :param svg_dict: dict {int: str} - ключ: номер, значение: SVG строка
    :param output_folder: str - путь к папке для сохранения .png файлов
    :param num_processes: int - количество процессов (по умолчанию: число ядер CPU)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tasks = [(key, svg_str, output_folder) for key, svg_str in svg_dict.items()]

    if num_processes is None:
        num_processes = cpu_count()

    print(f"Запуск обработки на {num_processes} процессах...")

    with Pool(processes=num_processes) as pool:
        results = pool.map(save_svg_to_png, tasks)
    return dict(results)

def save_svg_to_png(args):
    """
    Вспомогательная функция для multiprocessing.
    Принимает кортеж (key, svg_str, output_folder)
    """
    key, svg_str, output_folder = args
    output_path = os.path.join(output_folder, f"{key}.png")
    try:
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=output_path)
        return key, True
    except Exception as e:
        print(f"Ошибка при обработке {key}: {e}")
        return key, False

def svg_to_image_in_memory(svg_str):
    """
    Конвертирует SVG-строку в Image
    """
    svg_plug = '''
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="40" fill="red"/>
    </svg>
    '''
    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    except Exception as e:
        png_bytes = cairosvg.svg2png(bytestring=svg_plug.encode('utf-8'))
    image = Image.open(BytesIO(png_bytes))
    return image

def extract_svg_code(text):
    match_full = re.search(r"<svg\b[^>]*>.*?</svg>", text, re.DOTALL)
    if match_full:
        return (match_full.group(0),2)

    match_open = re.search(r"<svg\b[^>]*>.*", text, re.DOTALL)
    if match_open:
        return (match_open.group(0),1)

    return (None,0)
def convert_svg_to_png_in_memory(args):
    output_model, eval_func = args
    svg_text, res_flag = extract_svg_code(output_model)
    #here should be eval func to check validity svg_text
    image = svg_to_image_in_memory(svg_text)
    return (image, res_flag)
def eval_func():
    pass
def convert_svgs_to_pngs_parallel_in_memory(output_models_list, eval_func, num_processes=None):
    tasks = [(out,eval_func) for out in output_models_list]
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Запуск обработки на {num_processes} процессах...")
    with Pool(processes=num_processes) as pool:
        results = pool.map(convert_svg_to_png_in_memory, tasks)
    return results
if __name__ == '__main__':
    svg_dict = {
        i: '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
        for i  in range(32)
    }
    print(convert_svgs_to_pngs_parallel_in_memory(list(svg_dict.values()), eval_func))

    print(convert_svgs_to_pngs_parallel(svg_dict, 'output_images'))
