import customtkinter as ctk
import tkinter as tk
from Classifier import BasicClassifier
import easygui as ui



# Global Variables
train_path = ""
test_path = ""
app_name = "Micro Expressions Classifier"



# App Functions
def select_train_folder():
    global train_path
    train_path = ui.diropenbox(default="./")

def select_test_folder():
    global test_path
    test_path = ui.diropenbox(default="./")


def run_classifier():
    num_categories = int(num_cat_entry.get())
    split_size = img_size_x_entry.get().split(",")
    x, y = int(split_size[0]), int(split_size[1])
    batch_size = int(batch_num_entry.get())
    epochs = int(num_epochs_entry.get())
    try:
        classifier = BasicClassifier(num_categories, x, y, 3)
        classifier.get_data(train_path, test_path, batch_size)
        classifier.train(epochs)
        ui.msgbox("Training Complete", ok_button="Test Accuracy")
        loss, acc = classifier.evaluate()
        ui.msgbox(f'Model Accuracy= {acc} / Model Loss= {loss}', ok_button="Exit")

    except Exception as error:
        print(error)



# App
app = ctk.CTk()
app.title(app_name)
app.geometry("850x400")




# App Widgets
title_label = ctk.CTkLabel(app, text=app_name, font=("Arial", 40))
title_label.grid(row=0, column=1, padx=10, pady=50)

batch_num_entry = ctk.CTkEntry(app, placeholder_text="Batch Size")
batch_num_entry.grid(row=1, column=0, padx=10, pady=10)

img_size_x_entry = ctk.CTkEntry(app, placeholder_text="Image Size: X,Y")
img_size_x_entry.grid(row=1, column=1, padx=10, pady=10)

num_cat_entry = ctk.CTkEntry(app, placeholder_text="Number Of Different Categories")
num_cat_entry.grid(row=1, column=2, padx=10, pady=10)

num_epochs_entry = ctk.CTkEntry(app, placeholder_text="Number of Epochs")
num_epochs_entry.grid(row=2, column=0, padx=10, pady=10)

train_folder_button = ctk.CTkButton(app, text="Choose Train Folder", command=select_train_folder)
train_folder_button.grid(row=2, column=1, padx=10, pady=10)

test_folder_button = ctk.CTkButton(app, text="Choose Test Folder", command=select_test_folder)
test_folder_button.grid(row=2, column=2, padx=10, pady=10)



# Run Button
run_button = ctk.CTkButton(app, text="Run", command=run_classifier)
run_button.grid(row=3, column=1, padx=10, pady=50)



# Run App
app.mainloop()



