import io
import base64
from flask import Flask, request, render_template_string, send_file
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained models.
# For the colorization model, we pass a custom object for the loss function "mse".
era_classifier = tf.keras.models.load_model('era_classifier.h5')
colorization_model = tf.keras.models.load_model(
    'colorization_model.h5',
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

# Define the era class names (must match your training folders)
class_names = ['1900s', '1950s', '1970s']
num_classes = len(class_names)

def preprocess_for_classifier(image):
    """Resize image to 224x224, ensure 3 channels, and normalize to [0, 1]."""
    image = image.resize((224, 224))
    image_np = np.array(image)
    if image_np.ndim == 2:  # Grayscale: stack to create RGB
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[-1] == 1:  # Single channel: replicate to form RGB
        image_np = np.concatenate([image_np] * 3, axis=-1)
    image_np = image_np / 255.0
    return np.expand_dims(image_np, axis=0)

def preprocess_for_colorization(image):
    """Resize image to 256x256, convert to grayscale, and normalize to [0, 1]."""
    image = image.resize((256, 256))
    image_gray = image.convert('L')
    image_gray_np = np.array(image_gray) / 255.0
    return np.expand_dims(np.expand_dims(image_gray_np, axis=-1), axis=0)

# Enhanced HTML templates with Bootstrap and manual era selection
index_html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Historical Image Colorization</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
      function toggleEraSelection() {
        var eraOption = document.querySelector('input[name="era_option"]:checked').value;
        document.getElementById('manual-select').style.display = (eraOption === 'manual') ? 'block' : 'none';
      }
      window.onload = toggleEraSelection;
    </script>
  </head>
  <body class="bg-light">
    <div class="container mt-5">
      <h1>Historical Image Colorization</h1>
      <p>Upload a grayscale historical photo for era prediction and colorization.</p>
      <form action="/colorize" method="post" enctype="multipart/form-data">
        <div class="mb-3">
          <input class="form-control" type="file" name="file" accept="image/*" required>
        </div>
        <div class="mb-3">
          <label class="form-label">Era Option:</label><br>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="radio" name="era_option" id="auto" value="auto" checked onchange="toggleEraSelection()">
            <label class="form-check-label" for="auto">Auto Detect</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="radio" name="era_option" id="manual" value="manual" onchange="toggleEraSelection()">
            <label class="form-check-label" for="manual">Manual Selection</label>
          </div>
        </div>
        <div class="mb-3" id="manual-select" style="display:none;">
          <label for="selected_era" class="form-label">Select Era:</label>
          <select class="form-select" name="selected_era" id="selected_era">
            <option value="1900s">1900s</option>
            <option value="1950s">1950s</option>
            <option value="1970s">1970s</option>
          </select>
        </div>
        <button type="submit" class="btn btn-primary">Colorize</button>
      </form>
    </div>
  </body>
</html>
"""

result_html = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Colorization Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <div class="container mt-5">
      <h1>Colorization Result</h1>
      <p><strong>Selected Era:</strong> {{ era_label }}</p>
      <div class="mb-3">
        <img src="data:image/jpeg;base64,{{ img_str }}" class="img-fluid rounded" alt="Colorized Image">
      </div>
      <a href="/" class="btn btn-secondary">Try Another Image</a>
      <a href="/download?img={{ img_str }}" class="btn btn-success ms-2">Download Image</a>
    </div>
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(index_html)

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    try:
        image = Image.open(file).convert('RGB')
    except Exception as e:
        return 'Invalid image file.', 400

    # Check era option (auto or manual)
    era_option = request.form.get("era_option", "auto")
    if era_option == "manual":
        # Get selected era from form.
        selected_era = request.form.get("selected_era")
        if selected_era not in class_names:
            return "Invalid era selection.", 400
        era_index = class_names.index(selected_era)
        era_label = selected_era
    else:
        # Auto-detect using the era classifier.
        classifier_input = preprocess_for_classifier(image)
        era_pred = era_classifier.predict(classifier_input)
        era_index = int(np.argmax(era_pred, axis=1)[0])
        era_label = class_names[era_index]

    # Create a one-hot vector for the chosen era.
    era_vector = np.zeros((1, num_classes))
    era_vector[0, era_index] = 1

    # Preprocess image for colorization.
    colorization_input = preprocess_for_colorization(image)
    print("Colorization input range: min =", colorization_input.min(), "max =", colorization_input.max())
    
    # Predict colorized image.
    colorized_output = colorization_model.predict([colorization_input, era_vector])
    print("Model output range: min =", colorized_output.min(), "max =", colorized_output.max())
    
    # Process output: remove batch dimension and scale from [0, 1] to [0, 255].
    colorized_output = np.squeeze(colorized_output, axis=0)
    colorized_output = (colorized_output * 255).clip(0, 255).astype(np.uint8)
    colorized_image = Image.fromarray(colorized_output)

    # Encode result image to base64 for embedding.
    buffered = io.BytesIO()
    colorized_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template_string(result_html, era_label=era_label, img_str=img_str)

# Optional download endpoint.
@app.route('/download')
def download():
    img_b64 = request.args.get('img')
    if not img_b64:
        return "No image data.", 400
    img_data = base64.b64decode(img_b64)
    return send_file(io.BytesIO(img_data), mimetype='image/jpeg', as_attachment=True, download_name='colorized.jpg')

if __name__ == '__main__':
    app.run(debug=True)
