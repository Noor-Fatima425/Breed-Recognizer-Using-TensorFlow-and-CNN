# Breed-Recognizer-Using-TensorFlow-and-CNN
This project is a deep learning-based image classification system that identifies the breed of a cat or dog from an uploaded image using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

My pet cat's name is Toby and I've always wondered what breed he belonged to. According to the pet shop owner, he was a mixed breed Persian cat but this never convinced me.I tried Google Lens to find out about it and got different suggestions such as Ragdoll,British Shirthair etc..but Ragdoll was the most accurate for my cat. In order to clarify this matter, I came up with this idea to create a classifier model based on the Oxford-IIT Pet Dataset. Since it contained all the breed suggestions given by Google for Toby, it was ideal for my task.
# Version_1 classifier for both dog and cat breeds :
Analysis of Version1_both_cats_and_dogs.ipynb:
Dataset Handling
It loads the full dataset from trainval.txt.
It includes all breeds — both cats and dogs.
# Label Processing
df['breed_name'] = df['filename'].str.extract(r'^(.+)_\d+\.jpg')
df['class_id'] = df['breed_name'].astype('category').cat.codes
breed_name extraction is correct.

class_id uses all unique breed names.

id_to_breed dictionary will include both cat and dog breeds.

# Model Architecture
Standard CNN:

model = Sequential([
    Conv2D(32, ...),
    MaxPooling2D(...),
    Conv2D(64, ...),
    MaxPooling2D(...),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

This will output cat or dog breed names, depending on prediction.

predict_image_with_breed("data/images/toby.jpg")
(35, 'wheaten_terrier_')

# It unsuccessfully predicted my cat Toby as a dog breed :(
Imbalanced dataset is the reason!
If the training data has more dog breeds or more dog images than cat ones, the model may:
Learn dog features better
Be biased toward predicting dog classes

So this is what I did to predict my cat's breed. I created a version2 for my project to just deal with cat breeds...Yes i even thought of create a step-2 classifier model but it required to split the dataset anyway...which seemed more of a headache than simply filtering out the dogs...

Analysis of Version2_cats_breeds_classifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(12, activation='softmax')
])

cat_breeds = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
    "Russian_Blue", "Siamese", "Sphynx"
]

breed_to_id = {b: i for i, b in enumerate(cat_breeds)}
df['breed_name_cleaned'] = df['breed_name'].str.replace("_", "", regex=False)
cat_breeds_cleaned = [b.replace("_", "") for b in cat_breeds]
breed_name_lookup = dict(zip(cat_breeds_cleaned, cat_breeds))

df = df[df['breed_name_cleaned'].isin(cat_breeds_cleaned)].reset_index(drop=True)
df['breed_name'] = df['breed_name_cleaned'].map(breed_name_lookup)
df['class_id'] = df['breed_name'].map(breed_to_id)
id_to_breed = {i: b for i, b in enumerate(cat_breeds)}

This correctly predicted my cat's breed as Ragdoll.
# Yes! My catty is a Ragdoll :)
NOT a Persian!
