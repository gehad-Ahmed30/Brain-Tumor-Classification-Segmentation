import os
from flask import Flask, request, render_template , session
from datetime import timedelta
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from utils.predictor import predict_tumor_classification, predict_and_visualize_segmentation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = load_model("best_model.keras")

app.secret_key = "b9f@S!7m$2zW#xK4vLrDqP0nGuT3eXaY" 
app.permanent_session_lifetime = timedelta(minutes=1)  


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('image')
    
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['uploaded_image'] = filepath  
   
    elif 'uploaded_image' in session:
        filepath = session['uploaded_image']
        filename = os.path.basename(filepath)
    else:
        return render_template('index.html', result="Please upload an image first.")

    action = request.form.get("action")
    image_filename = os.path.basename(filepath)

    if action == "classify":
        result = predict_tumor_classification(filepath)
        return render_template(
            'index.html',
            result=result,
            image_path='uploads/' + image_filename
        )

    elif action == "segment":
        segmented_image_path = predict_and_visualize_segmentation(filepath)
        segmented_filename = os.path.basename(segmented_image_path)
        return render_template(
            'index.html',
            segmented_image='results/' + segmented_filename,
            image_path='uploads/' + image_filename
        )

    return render_template('index.html', result="Something went wrong.")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

