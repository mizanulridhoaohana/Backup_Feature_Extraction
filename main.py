import joblib
from PIL import Image, ImageOps
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte

def compute_glcm_features(image_path):
    img = io.imread(image_path)
    gray = color.rgb2gray(img)
    image = img_as_ubyte(gray)
    
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)

    max_value = inds.max() + 1
    matrix_cooccurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

    contrast = graycoprops(matrix_cooccurrence, 'contrast')
    dissimilarity = graycoprops(matrix_cooccurrence, 'dissimilarity')
    homogeneity = graycoprops(matrix_cooccurrence, 'homogeneity')
    energy = graycoprops(matrix_cooccurrence, 'energy')
    correlation = graycoprops(matrix_cooccurrence, 'correlation')
    asm = graycoprops(matrix_cooccurrence, 'ASM')

    return {
        "Contrast": contrast,
        "Dissimilarity": dissimilarity,
        "Homogeneity": homogeneity,
        "Energy": energy,
        "Correlation": correlation,
        "ASM": asm
    }


# Load the trained MLP classifier model from a file
model_filename = './model/mlp_classifier_model.joblib'
loaded_mlp_classifier = joblib.load(model_filename)

image_path = './data/NOCORROSION/100e35cf19.jpg'
target_size = (256, 256)

image = Image.open(image_path)

# Resize Image
image_new = ImageOps.fit(image, target_size, method=0, bleed=0.0, centering=(0.5, 0.5))
temp_image_path = './temp_resized_image.jpg'
image_new.save(temp_image_path)

# Extract GLCM Value
features = compute_glcm_features(temp_image_path)

# Initialize a new dictionary to store the transformed features
transformed_features = {}

# Define a list of angles
angles = ['0', '45', '90', '135']

# Iterate through the features and angles to create new labels
for feature_name, feature_values in features.items():
    for i, angle in enumerate(angles):
        new_label = f'{feature_name}{angle}'
        transformed_features[new_label] = feature_values[0][i]

# Transform the dictionary to a 1D NumPy array
transformed_features_array = np.array(list(transformed_features.values())).reshape(1, -1)

# Make predictions using the loaded model
predictions = loaded_mlp_classifier.predict(transformed_features_array)

if predictions[0] == 0:
    print("HASIL PREDIKSI: Corrosion")
else:
    print("HASIL PREDIKSI: NoCorrosion")

# Close and remove the temporary image file
image_new.close()
if os.path.exists(temp_image_path):
    os.remove(temp_image_path)