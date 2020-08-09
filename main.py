import tensorflow.keras
from PIL import Image, ImageOps, ImageTk
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog, Button, Label


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Add the classes labels
class_label = [
    'أ',
    'ب',
    'ت',
    'ث',
    'ج',
    'ح',
    'خ',
    'د',
    'ذ',
    'ر',
    'ز',
    'س',
    'ش',
    'ص',
    'ض',
    'ط',
    'ظ',
    'ع',
    'غ',
    'ف',
    'ق',
    'ك',
    'ل',
    'م',
    'ن',
    'ه',
    'و',
    'ي']


# Replace this with the path to your image
# image = Image.open(Path("test/id_1_label_1.png")).convert('RGB')


# The prediction function
def predictionfunc():
    image = Image.open(Path(browsefunc.file_name)).convert('RGB')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # print(prediction)

    # save the results in percentage form
    prediction_percentage = [x * 100 for x in prediction]
    # print(prediction_percentage)

    # find the max value
    # prediction_max = np.max(prediction_percentage)
    # print(prediction_max)

    # find the index of the max value and print the letter on the console
    predictionfunc.prediction_index = np.argmax(prediction_percentage)
    prediction_letter = class_label[predictionfunc.prediction_index]
    print(prediction_letter)


# Tkinter UI
window = Tk()

window.title("Handwritten Arabic Character Recognition")
window.geometry('550x550')


# The browse function
def browsefunc():
    browsefunc.file_name = filedialog.askopenfilename()
    predictionfunc()

    # Display the predicted letter
    letter = class_label[int(predictionfunc.prediction_index)]
    path_label.config(text=letter, font=("Courier", 100))

    # Display the selected image
    img = Image.open(Path(browsefunc.file_name))
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img


browse_button = Button(window, text="Browse", command=browsefunc)
browse_button.grid(column=2, row=3)
browse_button.config(font=("Courier", 25))

path_label = Label(window)
path_label.grid(column=5, row=5)

img_label = Label(window)
img_label.grid(column=20, row=20)

window.rowconfigure(5, weight=1)
window.columnconfigure(5, weight=1)

window.mainloop()
