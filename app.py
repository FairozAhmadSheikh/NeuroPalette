from flask import Flask, render_template, request, redirect, url_for
import os
from models.style_transfer import stylize_image
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content_file = request.files['content']
        style_file = request.files['style']

        content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))

        content_file.save(content_path)
        style_file.save(style_path)

        output_path = stylize_image(content_path, style_path)

        return render_template('index.html', content=content_path, style=style_path, output=output_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)