#!/usr/bin/env python3
"""
Flask API for PDF to CSV Parser
Provides REST endpoints for the Subscription Tracker.
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pdf_parser import PDFToCSVParser, BankStatementParser
from bank_parser_enhanced import EnhancedBankParser

app = Flask(__name__)
# Configure CORS to allow requests from your frontend
CORS(app, origins=["https://aarongreenberg.net", "http://localhost:3000", "https://*.vercel.app"], 
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'pdf-parser'})


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response


@app.route('/parse', methods=['POST', 'OPTIONS'])
def parse_pdf():
    """
    Parse a PDF file and return structured data.
    
    Expects multipart/form-data with:
    - file: PDF file
    - type: 'general' or 'bank_statement'
    """
    print(f"Received parse request with files: {request.files}")
    print(f"Form data: {request.form}")
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF and CSV files are allowed'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get parser type
        parser_type = request.form.get('type', 'general')
        
        if parser_type == 'bank_statement':
            # Use enhanced bank statement parser for better detection
            parser = EnhancedBankParser()
            results = parser.parse_file(filepath)
            
            print(f"Bank statement parsing results: {results}")
            
            # Handle both success and error cases
            if results.get('success', False):
                response = {
                    'success': True,
                    'type': 'subscriptions',
                    'data': {
                        'subscriptions': results.get('subscriptions', []),
                        'summary': {
                            'total_monthly': results.get('total_monthly', 0),
                            'total_annual': results.get('total_annual', 0),
                            'total_subscriptions': len(results.get('subscriptions', [])),
                            'transactions_analyzed': results.get('transactions_analyzed', 0)
                        },
                        'transactions': results.get('all_transactions', [])
                    }
                }
            else:
                # If no subscriptions found, still return success but with empty list
                response = {
                    'success': True,
                    'type': 'subscriptions',
                    'data': {
                        'subscriptions': [],
                        'summary': {
                            'total_monthly': 0,
                            'total_annual': 0,
                            'total_subscriptions': 0,
                            'transactions_analyzed': results.get('transactions_analyzed', 0)
                        }
                    },
                    'message': 'No subscriptions found in this statement'
                }
            
            print(f"Sending response: {response}")
        else:
            # Use general parser
            parser = PDFToCSVParser()
            results = parser.parse_pdf(filepath)
            
            response = {
                'success': results['success'],
                'type': 'general',
                'data': results['data'],
                'metadata': results['metadata']
            }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(f"Error processing PDF: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/parse-to-csv', methods=['POST'])
def parse_to_csv():
    """
    Parse PDF and return as CSV file.
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Parse and save to CSV
        parser = PDFToCSVParser()
        csv_path = filepath.replace('.pdf', '.csv')
        results = parser.parse_pdf(filepath, csv_path)
        
        if results['success'] and os.path.exists(csv_path):
            return send_file(
                csv_path,
                as_attachment=True,
                download_name=filename.replace('.pdf', '.csv'),
                mimetype='text/csv'
            )
        else:
            return jsonify({'error': 'Failed to parse PDF'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up files
        for path in [filepath, csv_path]:
            if os.path.exists(path):
                os.remove(path)


@app.route('/analyze-subscriptions', methods=['POST'])
def analyze_subscriptions():
    """
    Analyze bank statements to find all subscriptions.
    Accepts multiple PDF files.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    all_subscriptions = []
    
    for file in files:
        if not allowed_file(file.filename):
            continue
        
        # Save and process file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            parser = BankStatementParser()
            results = parser.find_subscriptions(filepath)
            
            if results['success']:
                all_subscriptions.extend(results['subscriptions'])
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    # Aggregate results
    unique_subscriptions = {}
    for sub in all_subscriptions:
        key = sub['name']
        if key not in unique_subscriptions:
            unique_subscriptions[key] = sub
        else:
            # Keep the most recent one
            if sub['date'] > unique_subscriptions[key]['date']:
                unique_subscriptions[key] = sub
    
    # Calculate totals
    total_monthly = sum(s['amount'] for s in unique_subscriptions.values() if s['frequency'] == 'monthly')
    total_annual = sum(s['amount'] for s in unique_subscriptions.values() if s['frequency'] == 'annual')
    
    return jsonify({
        'success': True,
        'subscriptions': list(unique_subscriptions.values()),
        'summary': {
            'total_subscriptions': len(unique_subscriptions),
            'total_monthly_cost': total_monthly,
            'total_annual_cost': total_annual,
            'estimated_yearly': total_monthly * 12 + total_annual
        }
    })


if __name__ == '__main__':
    # For production on Railway
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)