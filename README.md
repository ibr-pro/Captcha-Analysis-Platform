# Captcha Analysis Platform

A web-based platform for analyzing captcha difficulty using dual image processing algorithms. Upload captchas, test against two OCR methods, and receive automated difficulty assessments.

## Features

- **Dual Algorithm Testing**: EasyOCR and TrOCR (Transformer-based)
- **Batch Processing**: Analyze 5-15 captchas simultaneously
- **Automated Difficulty Classification**: EASY / MEDIUM / HARD based on algorithm performance
- **Real-time Progress Tracking**: Live updates during analysis
- **Comprehensive Reporting**: Accuracy metrics, processing times, and recommendations

## Installation

```bash
pip install flask easyocr transformers torch pillow werkzeug
```

## Quick Start

```bash
python app.py
```

Access the platform at `http://0.0.0.0:2020`

## Project Structure

```
captcha_platform/
├── app.py              # Flask backend server
├── templates/
│   └── index.html      # Frontend interface
└── uploads/            # Temporary storage (auto-created)
```

## Usage

1. Upload 5-15 captcha images
2. Enter correct answer for each captcha
3. Click "Analyze Captchas"
4. Review individual results and comprehensive report

## Difficulty Classification

| Level | Criteria |
|-------|----------|
| **EASY** | Method-1 accuracy ≥ 80% |
| **MEDIUM** | Method-1 < 80% AND Method-2 ≥ 70%, OR both methods 55-79% |
| **HARD** | Both methods < 55% |

## How It Works

1. **Image Processing Method-1**: Fast, good for standard captchas
2. **Image Processing Method-2**: Advanced AI for complex captchas
3. **Text Cleaning**: Removes non-alphanumeric characters before comparison
4. **Exact Match**: 100% accuracy if prediction matches user input, 0% otherwise


## Report Metrics

- **Accuracy**: Percentage of exact matches
- **Processing Time**: Average time per algorithm
- **Success Rate**: Successful predictions vs total
- **Recommendation**: Suggested approach based on results

## Requirements

- Python 3.7+
- 4GB+ RAM
- Internet connection (first run to download models)

## Supported Formats

PNG, JPG, JPEG, TIFF, GIF, BMP (max 16MB per file)

## Contributing

Pull requests are welcome. For major changes, please open an issue first.
