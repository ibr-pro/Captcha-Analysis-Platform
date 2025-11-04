from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import json
import time
import re
from datetime import datetime
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import shutil

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

reader = easyocr.Reader(['en'], gpu=False)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_prediction(text):
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def solve_captcha_1(image_path):
    try:
        start_time = time.time()
        result = reader.readtext(image_path)
        end_time = time.time()
        
        if result and len(result) > 0:
            text = result[0][-2]
            confidence = result[0][-1]
            processing_time = end_time - start_time
            cleaned_text = clean_prediction(text)
            return {
                'success': True,
                'text': text,
                'cleaned_text': cleaned_text,
                'confidence': confidence,
                'processing_time': processing_time
            }
        return {'success': False, 'error': 'No text detected'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def solve_captcha_2(image_path):
    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        end_time = time.time()
        
        processing_time = end_time - start_time
        cleaned_text = clean_prediction(generated_text)
        return {
            'success': True,
            'text': generated_text.strip(),
            'cleaned_text': cleaned_text,
            'processing_time': processing_time
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_image_accuracy(prediction, user_answer):
    if not prediction or not user_answer:
        return 0.0
    
    cleaned_prediction = clean_prediction(prediction)
    cleaned_answer = clean_prediction(user_answer)
    
    if cleaned_prediction == cleaned_answer:
        return 100.0
    else:
        return 0.0

def analyze_difficulty(results, user_answers):
    total_captchas = len(results)
    if total_captchas == 0:
        return {}
    
    method1_correct = 0
    method2_correct = 0
    method1_successful = 0
    method2_successful = 0
    
    for i, result in enumerate(results):
        user_answer = user_answers[i] if i < len(user_answers) else ""
        
        # Only count as successful if OCR completed without error
        if result['algorithm_1']['success']:
            method1_successful += 1
            # Only count as correct if prediction matches user answer
            accuracy = calculate_image_accuracy(result['algorithm_1']['cleaned_text'], user_answer)
            if accuracy == 100.0:
                method1_correct += 1
        
        if result['algorithm_2']['success']:
            method2_successful += 1
            accuracy = calculate_image_accuracy(result['algorithm_2']['cleaned_text'], user_answer)
            if accuracy == 100.0:
                method2_correct += 1
    
    # Calculate accuracy based on correct predictions, not just successful OCR
    method1_accuracy = (method1_correct / total_captchas) * 100 if total_captchas > 0 else 0
    method2_accuracy = (method2_correct / total_captchas) * 100 if total_captchas > 0 else 0
    
    # Calculate average processing time only for successful OCR attempts
    avg_processing_time_1 = 0
    avg_processing_time_2 = 0
    
    if method1_successful > 0:
        total_time_1 = sum(r['algorithm_1']['processing_time'] for r in results if r['algorithm_1']['success'])
        avg_processing_time_1 = total_time_1 / method1_successful
    
    if method2_successful > 0:
        total_time_2 = sum(r['algorithm_2']['processing_time'] for r in results if r['algorithm_2']['success'])
        avg_processing_time_2 = total_time_2 / method2_successful
    
    # Difficulty classification logic remains the same
    if method1_accuracy >= 80:
        difficulty = "EASY"
        recommendation = "Image Processing Method-1 works excellently for these captchas with high accuracy"
    elif method1_accuracy < 80 and method2_accuracy >= 70:
        difficulty = "MEDIUM"
        recommendation = "Image Processing Method-2 performs significantly better than Method-1 for these captchas"
    elif method1_accuracy >= 55 and method2_accuracy >= 55:
        difficulty = "MEDIUM"
        recommendation = "Both methods show moderate performance. Consider combining results or preprocessing images"
    elif method1_accuracy < 55 and method2_accuracy < 55:
        difficulty = "HARD"
        recommendation = "Both methods struggle significantly. Consider specialized preprocessing or alternative approaches"
    elif method1_accuracy >= 55 or method2_accuracy >= 55:
        if method1_accuracy > method2_accuracy:
            difficulty = "MEDIUM"
            recommendation = "Image Processing Method-1 shows better performance. Focus on optimizing this approach"
        else:
            difficulty = "MEDIUM"
            recommendation = "Image Processing Method-2 shows better performance. This method is more suitable"
    else:
        difficulty = "HARD"
        recommendation = "Captchas are challenging for automated recognition. Manual intervention may be required"
    
    return {
        'difficulty': difficulty,
        'method1_accuracy': round(method1_accuracy, 2),
        'method2_accuracy': round(method2_accuracy, 2),
        'method1_avg_time': round(avg_processing_time_1, 3),
        'method2_avg_time': round(avg_processing_time_2, 3),
        'total_captchas': total_captchas,
        'method1_successful': method1_successful,  # Count of successful OCR attempts
        'method2_successful': method2_successful,  # Count of successful OCR attempts
        'method1_correct': method1_correct,        # Count of truly correct predictions
        'method2_correct': method2_correct,        # Count of truly correct predictions
        'recommendation': recommendation,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_single', methods=['POST'])
def analyze_single():
    if 'captcha' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['captcha']
    user_answer = request.form.get('user_answer', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            result1 = solve_captcha_1(file_path)
            result2 = solve_captcha_2(file_path)
            
            method1_accuracy = 0
            method2_accuracy = 0
            
            if result1['success']:
                method1_accuracy = calculate_image_accuracy(result1['cleaned_text'], user_answer)
            
            if result2['success']:
                method2_accuracy = calculate_image_accuracy(result2['cleaned_text'], user_answer)
            
            result = {
                'algorithm_1': result1,
                'algorithm_2': result2,
                'method1_accuracy': method1_accuracy,
                'method2_accuracy': method2_accuracy,
                'user_answer': user_answer,
                'filename': filename
            }
            
            os.remove(file_path)
            return jsonify(result)
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.get_json()
    results = data.get('results', [])
    user_answers = data.get('user_answers', [])
    
    if not results:
        return jsonify({'error': 'No results provided'}), 400
    
    report = analyze_difficulty(results, user_answers)
    return jsonify(report)

if __name__ == '__main__':
    print("Starting Captcha Analysis Platform...")
    print("Server will be available at: http://0.0.0.0:2020")
    app.run(host='0.0.0.0', port=2020, debug=False)