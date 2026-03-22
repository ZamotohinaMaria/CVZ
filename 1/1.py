import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def lsb_embed(C: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """
    Стеганографическое НЗБ-встраивание в первую битовую плоскость
    двоичного вектора b в контейнер C.
    
    Параметры:
    C: ndarray - контейнер (изображение в градациях серого)
    b: ndarray - двоичный вектор для встраивания
    seed: int - определяет порядок записи
        seed < 0: случайный порядок
        seed >= 0: последовательный порядок (seed игнорируется или используется для генерации фиксированного случайного порядка если seed > 0)
    """
    Cw = C.copy()
    rows, cols = C.shape
    total_pixels = rows * cols
    Nb = len(b)
    
    # Если вектор слишком длинный для контейнера
    if Nb > total_pixels:
        raise ValueError(f"Длина вектора b ({Nb}) превышает размер контейнера ({total_pixels})")
    
    # Генерируем порядок встраивания
    if seed < 0:
        # Случайный порядок
        indices = random.sample(range(total_pixels), Nb)
    else:
        # Последовательный порядок
        indices = list(range(Nb))
    
    # Встраиваем биты
    for i, idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        
        # Заменяем младший бит
        Cw[row, col] = (Cw[row, col] & 0xFE) | b[i]
    
    return Cw

def lsb_extract(Cw: np.ndarray, Nb: int, seed: int) -> np.ndarray:
    """
    Извлечение двоичного вектора длины Nb, встроенного в контейнер Cw.
    
    Параметры:
    Cw: ndarray - контейнер со встроенным сообщением
    Nb: int - длина извлекаемого вектора
    seed: int - должен соответствовать seed, использованному при встраивании
    """
    rows, cols = Cw.shape
    total_pixels = rows * cols
    
    if Nb > total_pixels:
        raise ValueError(f"Запрашиваемая длина вектора ({Nb}) превышает размер контейнера ({total_pixels})")
    
    # Генерируем порядок извлечения (должен совпадать с порядком встраивания)
    if seed < 0:
        # Случайный порядок
        random.seed(abs(seed))
        indices = random.sample(range(total_pixels), Nb)
    else:
        # Последовательный порядок
        indices = list(range(Nb))
    
    # Извлекаем биты
    b_extracted = np.zeros(Nb, dtype=np.uint8)
    
    for i, idx in enumerate(indices):
        row = idx // cols
        col = idx % cols
        
        # Извлекаем младший бит
        b_extracted[i] = Cw[row, col] & 0x01
    
    return b_extracted

def plusminus_embed(C: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """
    ±1-встраивание через вызов функции lsb_embed.
    """
    # Преобразуем биты {0,1} в знаки {-1,1}
    b_signs = 2 * b - 1
    
    # Встраиваем знаки
    Cw = lsb_embed(C, b_signs, seed)
    
    return Cw

def compare_bitwise(original: np.ndarray, extracted: np.ndarray) -> None:
    """
    Побитовое сравнение двух двоичных векторов.
    """
    if len(original) != len(extracted):
        print(f"Векторы разной длины: оригинальный {len(original)}, извлечённый {len(extracted)}")
        return
    
    matches = np.sum(original == extracted)
    total = len(original)
    accuracy = matches / total * 100
    
    print(f"Совпадение битов: {matches}/{total} ({accuracy:.2f}%)")
    
    # Выводим несовпадающие биты
    # if matches != total:
    #     print("Несовпадающие позиции:")
    #     for i in range(total):
    #         if original[i] != extracted[i]:
    #             print(f"  Позиция {i}: оригинал={original[i]}, извлечено={extracted[i]}")

def visualize_bit_plane(image: np.ndarray, title: str) -> None:
    """
    Визуализация первой битовой плоскости изображения.
    """
    # Извлекаем младший бит
    bit_plane = image & 0x01
    
    plt.figure(figsize=(8, 8))
    plt.imshow(bit_plane, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.show()

# Основной скрипт steglsb_run
def steglsb_run():
    """
    Основной скрипт для выполнения всех заданий.
    """
    print("=== Реализация и исследование систем стеганографического НЗБ-встраивания ===\n")
    
    # 1. Загрузка тестового изображения
    print("1. Загрузка и подготовка тестового изображения...")
    # Создаём тестовое изображение 100x100
    C = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    print(f"   Размер контейнера: {C.shape}, общее количество пикселей: {C.size}")
    
    # 2. Генерация тестовой битовой строки
    print("\n2. Генерация тестовой битовой строки...")
    # Длина строки - 20% от объёма битовой плоскости
    Nb = int(C.size * 0.2)
    b_original = np.random.randint(0, 2, Nb, dtype=np.uint8)
    print(f"   Длина битовой строки: {Nb} бит")
    print(f"   Пример первых 10 битов: {b_original[:10]}")
    
    # 3. Тестирование lsb_embed (seed < 0)
    print("\n3. Тестирование lsb_embed с seed < 0 (случайный порядок)...")
    seed_random = -42
    Cw_random = lsb_embed(C, b_original, seed_random)
    print(f"   Контейнер со встроенным сообщением создан")
    print(f"   Изменено пикселей: {np.sum(C != Cw_random)}")
    
    # 4. Тестирование lsb_extract
    print("\n4. Тестирование lsb_extract...")
    b_extracted_random = lsb_extract(Cw_random, Nb, seed_random)
    compare_bitwise(b_original, b_extracted_random)
    
    # 5. Тестирование с seed >= 0 (последовательный порядок)
    print("\n5. Тестирование с seed >= 0 (последовательный порядок)...")
    seed_sequential = 0
    Cw_sequential = lsb_embed(C, b_original, seed_sequential)
    b_extracted_sequential = lsb_extract(Cw_sequential, Nb, seed_sequential)
    compare_bitwise(b_original, b_extracted_sequential)
    
    # 6. Визуализация битовых плоскостей
    print("\n6. Визуализация битовых плоскостей...")
    visualize_bit_plane(C, "Оригинальное изображение (младший бит)")
    visualize_bit_plane(Cw_random, "Изображение со случайным встраиванием (младший бит)")
    visualize_bit_plane(Cw_sequential, "Изображение с последовательным встраиванием (младший бит)")
    
    # 7. Тестирование plusminus_embed
    print("\n7. Тестирование plusminus_embed...")
    Cw_plusminus = plusminus_embed(C, b_original, seed_random)
    b_extracted_plusminus = lsb_extract(Cw_plusminus, Nb, seed_random)
    
    # Преобразуем обратно из {-1,1} в {0,1}
    b_extracted_plusminus = (b_extracted_plusminus + 1) // 2
    compare_bitwise(b_original, b_extracted_plusminus)
    
    # 8. Сравнение искажений
    print("\n8. Сравнение искажений контейнеров...")
    mse_random = np.mean((C.astype(float) - Cw_random.astype(float))**2)
    mse_sequential = np.mean((C.astype(float) - Cw_sequential.astype(float))**2)
    mse_plusminus = np.mean((C.astype(float) - Cw_plusminus.astype(float))**2)
    
    print(f"   MSE случайное встраивание: {mse_random:.6f}")
    print(f"   MSE последовательное встраивание: {mse_sequential:.6f}")
    print(f"   MSE ±1-встраивание: {mse_plusminus:.6f}")
    
    # 9. Проверка на реальном изображении
    print("\n9. Демонстрация на реальном изображении...")
    try:
        # Используем встроенное тестовое изображение из OpenCV
        real_image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if real_image is None:
            # Если файл не найден, создаём тестовое
            print("   Файл 'lena.png' не найден, создаём тестовое изображение...")
            real_image = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
            if real_image is None:
                real_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    except:
        real_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # Обрезаем для ускорения
    real_image = real_image[:128, :128]
    
    # Встраиваем сообщение
    Nb_real = int(real_image.size * 0.2)
    b_real = np.random.randint(0, 2, Nb_real, dtype=np.uint8)
    real_image_stego = lsb_embed(real_image, b_real, seed_random)
    
    # Визуализация
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(real_image, cmap='gray')
    plt.title('Оригинальное изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(real_image_stego, cmap='gray')
    plt.title('Стего-изображение')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(real_image.astype(float) - real_image_stego.astype(float)), cmap='hot')
    plt.title('Разность изображений')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Выполнение завершено ===")

if __name__ == "__main__":
    steglsb_run()