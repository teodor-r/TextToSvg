import sys
import pkgutil

print(f"Python path: {sys.path}")

# Проверка наличия пакета в системных модулях
if 'tifascore' in sys.modules:
    print("tifascore уже загружен")
else:
    print("tifascore не в sys.modules")

# Проверка доступности через pkgutil
spec = pkgutil.find_loader("tifascore")
if spec:
    print(f"Пакет найден: {spec.path}")
    try:
        from tifascore import *
        print("Импорт успешен!")
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
else:
    print("Пакет tifascore не найден в системе")