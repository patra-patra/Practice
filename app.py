import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import model_

# Глобальная переменная для пути к файлу
file_path2 = ""

def open_image():
    global file_path2  # Объявляем переменную как глобальную
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Изменяем размер изображения
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        recognize_btn.pack(pady=10)  # Показываем кнопку "Распознать"
        file_path2 = file_path  # Обновляем значение file_path2

def recognize_image():
    global file_path2  # Объявляем переменную как глобальную
    class_ = model_.result(file_path2)
    label_result.config(text=class_)

# Создаем главное окно
root = tk.Tk()
root.title("Image Viewer")

# Создаем кнопку для открытия изображения
btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack(pady=20)

# Панель для отображения изображения
panel = tk.Label(root)
panel.pack(pady=20)

# Создаем кнопку "Распознать"
recognize_btn = tk.Button(root, text="Распознать", command=recognize_image)
recognize_btn.pack_forget()  # Изначально скрываем кнопку

# Создаем метку для отображения результата
label_result = tk.Label(root, text="")
label_result.pack(pady=20)

# Запускаем главное окно
root.mainloop()