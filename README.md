# PDF to CSV Parser API

High-accuracy PDF and CSV parser for bank statements with automatic subscription detection.

## Features

- **95%+ Accuracy**: Multiple parsing strategies including OCR fallback
- **PDF Support**: Handles various PDF formats and bank statements
- **CSV Support**: Processes CSV exports from banks
- **Subscription Detection**: Automatically identifies recurring charges
- **Smart Categorization**: Categorizes subscriptions (streaming, software, fitness, etc.)
- **Confidence Scoring**: Rates detection confidence (high/medium/low)
- **RESTful API**: Easy integration with any frontend

## Tech Stack

- Python 3.11
- Flask (REST API)
- pdfplumber & PyPDF2 (PDF parsing)
- pytesseract (OCR for scanned PDFs)
- pandas (Data processing)
- Docker ready

## API Endpoints

### Health Check
```
GET /health
```

### Parse Bank Statement
```
POST /parse
Content-Type: multipart/form-data

Fields:
- file: PDF or CSV file
- type: 'bank_statement' or 'general'
```

Returns:
```json
{
  "success": true,
  "type": "subscriptions",
  "data": {
    "subscriptions": [...],
    "summary": {
      "total_monthly": 123.45,
      "total_annual": 99.99,
      "total_subscriptions": 5,
      "transactions_analyzed": 150
    },
    "transactions": [...]
  }
}
```

## Deployment

### Railway
The app is configured for Railway deployment with automatic builds.

### Local Development
```bash
pip install -r requirements.txt
python api.py
```

## Environment Variables

- `PORT`: Server port (default: 5001)

## License

Private - Part of the AMG Toolkit