#!/usr/bin/env python3
"""
Enhanced Bank Statement Parser with improved subscription detection.
Supports both PDF and CSV files.
"""

import os
import re
import csv
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import PyPDF2
    import pdfplumber
    import pandas as pd
    HAS_PDF_LIBS = True
except ImportError as e:
    logger.warning(f"Some PDF libraries are missing: {e}")
    HAS_PDF_LIBS = False


class EnhancedBankParser:
    """
    Enhanced parser for bank statements with better subscription detection.
    """
    
    def __init__(self):
        # Expanded subscription keywords
        self.subscription_keywords = [
            # Streaming Services
            'netflix', 'spotify', 'amazon prime', 'prime video', 'hulu', 'disney',
            'disney+', 'disneyplus', 'hbo', 'hbo max', 'paramount', 'peacock',
            'apple tv', 'apple music', 'youtube premium', 'youtube tv', 'twitch',
            'crunchyroll', 'audible', 'kindle unlimited', 'scribd',
            
            # Software & Productivity
            'adobe', 'microsoft', 'office 365', 'google workspace', 'google storage',
            'dropbox', 'icloud', 'github', 'gitlab', 'atlassian', 'jira', 'slack',
            'zoom', 'canva', 'figma', 'notion', 'evernote', 'todoist', 'monday.com',
            'clickup', 'asana', 'trello', 'airtable', 'basecamp',
            
            # AI & Tech Services
            'chatgpt', 'openai', 'claude', 'anthropic', 'midjourney', 'copilot',
            'aws', 'amazon web', 'digital ocean', 'heroku', 'vercel', 'netlify',
            
            # Security & Privacy
            'nordvpn', 'expressvpn', 'surfshark', 'lastpass', '1password',
            'dashlane', 'bitwarden', 'protonmail', 'proton vpn',
            
            # Education & Learning
            'skillshare', 'masterclass', 'coursera', 'udemy', 'pluralsight',
            'linkedin learning', 'duolingo', 'babbel', 'rosetta stone',
            
            # Fitness & Wellness
            'gym', 'fitness', 'planet fitness', '24 hour', 'anytime fitness',
            'peloton', 'strava', 'zwift', 'headspace', 'calm', 'noom',
            'weight watchers', 'ww digital', 'myfitnesspal',
            
            # Gaming
            'xbox', 'xbox game pass', 'playstation', 'ps plus', 'nintendo',
            'steam', 'epic games', 'discord nitro', 'twitch prime',
            
            # Food Delivery & Meal Kits
            'doordash', 'dashpass', 'uber eats', 'grubhub', 'postmates',
            'hellofresh', 'blue apron', 'factor', 'freshly',
            
            # News & Publications
            'new york times', 'nytimes', 'wall street journal', 'wsj',
            'washington post', 'economist', 'bloomberg', 'medium membership',
            'substack', 'patreon', 'onlyfans',
            
            # Other Services
            'subscription', 'membership', 'premium', 'pro', 'plus', 'annual',
            'monthly', 'recurring', 'autopay', 'automatic payment'
        ]
        
        # Common subscription amounts (including tax variations)
        self.common_amounts = {
            # Base prices
            4.99, 5.99, 6.99, 7.99, 8.99, 9.99, 10.99, 11.99, 12.99, 13.99,
            14.99, 15.99, 16.99, 17.99, 18.99, 19.99, 20.99, 21.99, 22.99,
            24.99, 29.99, 34.99, 39.99, 44.99, 49.99, 54.99, 59.99, 69.99,
            79.99, 89.99, 99.99, 119.99, 139.99, 149.99, 199.99,
            
            # Common after-tax amounts (roughly 10% tax)
            5.49, 6.59, 7.69, 8.79, 9.89, 10.99, 12.09, 13.19, 14.29,
            16.49, 21.99, 27.49, 32.99, 43.99, 54.99, 65.99, 87.99, 109.99
        }
    
    def parse_file(self, file_path: str) -> Dict:
        """Parse bank statement from PDF or CSV file."""
        if file_path.lower().endswith('.csv'):
            return self.parse_csv(file_path)
        elif file_path.lower().endswith('.pdf'):
            return self.parse_pdf(file_path)
        else:
            return {'success': False, 'error': 'Unsupported file type'}
    
    def parse_csv(self, csv_path: str) -> Dict:
        """Parse CSV bank statement."""
        try:
            transactions = []
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            # Try different encodings
            for encoding in encodings:
                try:
                    # First, detect the delimiter
                    with open(csv_path, 'r', encoding=encoding) as f:
                        sample = f.read(2048)
                        delimiter = ','
                        if '\t' in sample:
                            delimiter = '\t'
                        elif ';' in sample:
                            delimiter = ';'
                    
                    # Read the CSV
                    df = pd.read_csv(csv_path, encoding=encoding, delimiter=delimiter)
                    
                    # Process each row
                    for _, row in df.iterrows():
                        transaction = self._process_csv_row(row)
                        if transaction:
                            transactions.append(transaction)
                    
                    break  # Success
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Failed with encoding {encoding}: {e}")
                    continue
            
            # Find subscriptions
            subscriptions = self._identify_subscriptions(transactions)
            
            return {
                'success': True,
                'subscriptions': subscriptions,
                'total_monthly': sum(s['amount'] for s in subscriptions if s['frequency'] == 'monthly'),
                'total_annual': sum(s['amount'] for s in subscriptions if s['frequency'] == 'annual'),
                'transactions_analyzed': len(transactions)
            }
            
        except Exception as e:
            logger.error(f"CSV parsing failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def parse_pdf(self, pdf_path: str) -> Dict:
        """Parse PDF bank statement."""
        if not HAS_PDF_LIBS:
            return {'success': False, 'error': 'PDF libraries not installed'}
        
        try:
            transactions = []
            
            # Try pdfplumber first (better for tables)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        # Extract tables
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                transaction = self._process_table_row(row)
                                if transaction:
                                    transactions.append(transaction)
                        
                        # Also extract text for non-table content
                        text = page.extract_text()
                        text_transactions = self._extract_from_text(text)
                        transactions.extend(text_transactions)
            
            except Exception as e:
                logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
                
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        text_transactions = self._extract_from_text(text)
                        transactions.extend(text_transactions)
            
            # Remove duplicates
            unique_transactions = self._deduplicate_transactions(transactions)
            
            # Find subscriptions
            subscriptions = self._identify_subscriptions(unique_transactions)
            
            return {
                'success': True,
                'subscriptions': subscriptions,
                'total_monthly': sum(s['amount'] for s in subscriptions if s['frequency'] == 'monthly'),
                'total_annual': sum(s['amount'] for s in subscriptions if s['frequency'] == 'annual'),
                'transactions_analyzed': len(unique_transactions)
            }
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_csv_row(self, row: pd.Series) -> Optional[Dict]:
        """Process a CSV row into a transaction."""
        transaction = {}
        
        # Common field mappings
        date_fields = ['date', 'transaction date', 'trans date', 'posting date', 'value date']
        desc_fields = ['description', 'desc', 'narration', 'details', 'merchant', 'narrative', 'particulars']
        amount_fields = ['amount', 'debit', 'withdrawal', 'charge', 'payment']
        credit_fields = ['credit', 'deposit']
        
        # Extract date
        for field in date_fields:
            for col in row.index:
                if field in col.lower():
                    val = row[col]
                    if pd.notna(val):
                        transaction['date'] = str(val)
                        break
        
        # Extract description
        for field in desc_fields:
            for col in row.index:
                if field in col.lower():
                    val = row[col]
                    if pd.notna(val):
                        transaction['description'] = str(val)
                        break
        
        # Extract amount (handle both debit and credit)
        amount = None
        for field in amount_fields:
            for col in row.index:
                if field in col.lower():
                    val = row[col]
                    if pd.notna(val):
                        try:
                            amount = abs(float(str(val).replace('$', '').replace(',', '').replace('(', '').replace(')', '')))
                            transaction['amount'] = amount
                            break
                        except:
                            pass
        
        # Check credit columns (might be income)
        if not amount:
            for field in credit_fields:
                for col in row.index:
                    if field in col.lower():
                        val = row[col]
                        if pd.notna(val):
                            try:
                                amount = abs(float(str(val).replace('$', '').replace(',', '').replace('(', '').replace(')', '')))
                                transaction['amount'] = amount
                                transaction['type'] = 'credit'
                                break
                            except:
                                pass
        
        # Return transaction if it has essential fields
        if transaction.get('description') and transaction.get('amount'):
            return transaction
        
        return None
    
    def _process_table_row(self, row: List) -> Optional[Dict]:
        """Process a table row from PDF."""
        if not row or len(row) < 2:
            return None
        
        transaction = {}
        
        # Clean row values
        row = [str(cell).strip() if cell else '' for cell in row]
        
        # Look for date pattern
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        for i, cell in enumerate(row):
            if re.search(date_pattern, cell):
                transaction['date'] = cell
                # Description is usually after date
                if i + 1 < len(row):
                    transaction['description'] = row[i + 1]
                break
        
        # Look for amount pattern
        amount_pattern = r'[\$\-\+]?\d+[,\.]?\d*\.?\d{0,2}'
        amounts = []
        for cell in row:
            matches = re.findall(amount_pattern, cell)
            for match in matches:
                try:
                    amt = float(match.replace('$', '').replace(',', ''))
                    if amt > 0:
                        amounts.append(amt)
                except:
                    pass
        
        if amounts:
            transaction['amount'] = amounts[0]  # Usually first amount is transaction
        
        # If no description found, use the longest text field
        if 'description' not in transaction:
            texts = [cell for cell in row if len(cell) > 5 and not re.match(r'^[\d\$\.\-\+,]+$', cell)]
            if texts:
                transaction['description'] = max(texts, key=len)
        
        return transaction if transaction.get('description') and transaction.get('amount') else None
    
    def _extract_from_text(self, text: str) -> List[Dict]:
        """Extract transactions from unstructured text."""
        transactions = []
        lines = text.split('\n')
        
        # Common patterns in bank statements
        patterns = [
            # Date Description Amount
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)\s+([\$\-]?\d+[,\.]?\d*\.?\d{0,2})',
            # Description Date Amount
            r'(.+?)\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+([\$\-]?\d+[,\.]?\d*\.?\d{0,2})',
            # Multi-line with amount on next line
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)$'
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in patterns[:2]:  # First two are complete patterns
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    transaction = {}
                    
                    # Determine field order based on pattern
                    if re.match(r'\d{1,2}[-/]\d{1,2}', groups[0]):
                        transaction['date'] = groups[0]
                        transaction['description'] = groups[1].strip()
                    else:
                        transaction['description'] = groups[0].strip()
                        transaction['date'] = groups[1]
                    
                    # Extract amount
                    try:
                        transaction['amount'] = abs(float(groups[2].replace('$', '').replace(',', '')))
                    except:
                        continue
                    
                    if transaction.get('description') and transaction.get('amount'):
                        transactions.append(transaction)
                    break
            
            # Try multi-line pattern
            if not any(t.get('description', '').lower() in line.lower() for t in transactions):
                date_match = re.match(patterns[2], line)
                if date_match and i + 1 < len(lines):
                    # Check next line for amount
                    next_line = lines[i + 1].strip()
                    amount_match = re.search(r'[\$\-]?\d+[,\.]?\d*\.?\d{0,2}', next_line)
                    if amount_match:
                        try:
                            transaction = {
                                'date': date_match.group(1),
                                'description': date_match.group(2).strip(),
                                'amount': abs(float(amount_match.group().replace('$', '').replace(',', '')))
                            }
                            transactions.append(transaction)
                        except:
                            pass
        
        return transactions
    
    def _deduplicate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Remove duplicate transactions."""
        seen = set()
        unique = []
        
        for t in transactions:
            # Create a unique key
            key = f"{t.get('date', '')}|{t.get('description', '')}|{t.get('amount', 0)}"
            if key not in seen:
                seen.add(key)
                unique.append(t)
        
        return unique
    
    def _identify_subscriptions(self, transactions: List[Dict]) -> List[Dict]:
        """Enhanced subscription identification."""
        subscriptions = []
        merchant_history = {}  # Track all transactions by merchant
        
        # Group transactions by merchant
        for transaction in transactions:
            desc = transaction.get('description', '').lower()
            if not desc:
                continue
            
            merchant = self._extract_merchant_name(desc)
            amount = float(transaction.get('amount', 0))
            
            if merchant not in merchant_history:
                merchant_history[merchant] = []
            
            merchant_history[merchant].append({
                'amount': amount,
                'date': transaction.get('date', ''),
                'description': desc,
                'original': transaction
            })
        
        # Analyze each merchant
        for merchant, history in merchant_history.items():
            if not merchant:
                continue
            
            # Skip if only one transaction (unless it matches keywords)
            if len(history) == 1:
                desc = history[0]['description']
                amount = history[0]['amount']
                
                # Check if it's a known subscription
                is_keyword_match = any(kw in desc for kw in self.subscription_keywords)
                is_common_amount = self._is_common_subscription_amount(amount)
                
                if not (is_keyword_match or is_common_amount):
                    continue
            
            # Analyze transaction patterns
            amounts = [h['amount'] for h in history]
            unique_amounts = set(amounts)
            
            # Check for subscription patterns
            is_subscription = False
            confidence = 'low'
            frequency = 'monthly'
            subscription_amount = 0
            
            # Pattern 1: Keyword match
            descriptions = [h['description'] for h in history]
            if any(any(kw in desc for kw in self.subscription_keywords) for desc in descriptions):
                is_subscription = True
                confidence = 'high'
                subscription_amount = self._get_most_common_amount(amounts)
            
            # Pattern 2: Recurring same amount
            elif len(unique_amounts) <= 2 and len(history) >= 2:
                # Allow for small variations (tax, fees)
                amount_variance = max(unique_amounts) - min(unique_amounts)
                avg_amount = sum(amounts) / len(amounts)
                
                if amount_variance < avg_amount * 0.1:  # Less than 10% variance
                    is_subscription = True
                    confidence = 'medium' if len(history) >= 3 else 'low'
                    subscription_amount = avg_amount
            
            # Pattern 3: Common subscription amount
            elif any(self._is_common_subscription_amount(amt) for amt in amounts):
                is_subscription = True
                confidence = 'medium'
                subscription_amount = self._get_most_common_amount(amounts)
            
            # Pattern 4: Regular interval (e.g., monthly)
            elif self._has_regular_interval(history):
                is_subscription = True
                confidence = 'medium'
                subscription_amount = self._get_most_common_amount(amounts)
            
            if is_subscription:
                # Determine frequency
                if subscription_amount > 100:
                    frequency = 'annual'
                elif subscription_amount > 50:
                    frequency = 'quarterly' if 'quarter' in descriptions[0] else 'monthly'
                else:
                    frequency = 'monthly'
                
                # Get most recent transaction
                latest = max(history, key=lambda x: x['date'] if x['date'] else '')
                
                subscription = {
                    'name': merchant.title(),
                    'amount': round(subscription_amount, 2),
                    'date': latest['date'],
                    'frequency': frequency,
                    'category': self._categorize_subscription(latest['description']),
                    'confidence': confidence,
                    'occurrences': len(history)
                }
                
                subscriptions.append(subscription)
        
        # Sort by amount (highest first)
        subscriptions.sort(key=lambda x: x['amount'], reverse=True)
        
        return subscriptions
    
    def _extract_merchant_name(self, description: str) -> str:
        """Extract merchant name from transaction description."""
        desc = description.lower()
        
        # Remove common prefixes
        prefixes = [
            'pos debit', 'pos purchase', 'debit card purchase', 'debit purchase',
            'recurring payment', 'automatic payment', 'preauthorized',
            'web payment', 'online payment', 'card purchase', 'purchase',
            'direct debit', 'standing order', 'dd', 'so', 'ach',
            'visa purchase', 'mastercard purchase', 'amex purchase',
            'checkcard', 'debit card', 'credit card', 'e-payment'
        ]
        
        for prefix in prefixes:
            desc = re.sub(f'^{prefix}\\s*[-:]?\\s*', '', desc, flags=re.IGNORECASE)
        
        # Remove trailing transaction info
        desc = re.sub(r'\s+\*+\d+', '', desc)  # *1234
        desc = re.sub(r'\s+#\d+', '', desc)    # #1234
        desc = re.sub(r'\s+\d{10,}', '', desc)  # Long reference numbers
        desc = re.sub(r'\s+\d{2}/\d{2}', '', desc)  # Dates
        desc = re.sub(r'\s+card\s*\d+', '', desc)   # Card numbers
        
        # Extract URL if present
        url_match = re.search(r'([a-z0-9\-]+\.(com|net|org|io|co|app))', desc)
        if url_match:
            return url_match.group(1).split('.')[0]
        
        # Clean up and extract meaningful words
        words = desc.split()
        merchant_words = []
        
        for word in words:
            # Skip short words and numbers
            if len(word) <= 2 or word.isdigit():
                continue
            
            # Skip common words
            if word in ['the', 'and', 'for', 'inc', 'llc', 'ltd', 'corp']:
                continue
            
            # Clean and add word
            word = re.sub(r'[^a-z0-9]', '', word)
            if word:
                merchant_words.append(word)
            
            # Usually first 2-3 words are the merchant name
            if len(merchant_words) >= 2:
                break
        
        return ' '.join(merchant_words)
    
    def _is_common_subscription_amount(self, amount: float) -> bool:
        """Check if amount matches common subscription prices."""
        # Check exact matches
        if amount in self.common_amounts:
            return True
        
        # Check with small tolerance (for tax/fees)
        for common in self.common_amounts:
            if abs(amount - common) < 0.50:  # Within 50 cents
                return True
        
        return False
    
    def _get_most_common_amount(self, amounts: List[float]) -> float:
        """Get the most common amount from a list."""
        if not amounts:
            return 0
        
        # Count occurrences
        amount_counts = {}
        for amt in amounts:
            # Round to nearest cent
            amt = round(amt, 2)
            amount_counts[amt] = amount_counts.get(amt, 0) + 1
        
        # Return most common
        return max(amount_counts, key=amount_counts.get)
    
    def _has_regular_interval(self, history: List[Dict]) -> bool:
        """Check if transactions occur at regular intervals."""
        if len(history) < 3:
            return False
        
        # Sort by date
        try:
            sorted_history = sorted(history, key=lambda x: datetime.strptime(x['date'], '%m/%d/%Y'))
        except:
            return False  # Can't parse dates
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(sorted_history)):
            try:
                date1 = datetime.strptime(sorted_history[i-1]['date'], '%m/%d/%Y')
                date2 = datetime.strptime(sorted_history[i]['date'], '%m/%d/%Y')
                interval = (date2 - date1).days
                intervals.append(interval)
            except:
                pass
        
        if not intervals:
            return False
        
        # Check if intervals are regular (monthly = ~30 days)
        avg_interval = sum(intervals) / len(intervals)
        
        # Monthly subscription
        if 25 <= avg_interval <= 35:
            return all(25 <= i <= 35 for i in intervals)
        
        # Annual subscription
        if 350 <= avg_interval <= 380:
            return True
        
        return False
    
    def _categorize_subscription(self, description: str) -> str:
        """Categorize subscription based on description."""
        desc_lower = description.lower()
        
        categories = {
            'streaming': ['netflix', 'hulu', 'disney', 'hbo', 'paramount', 'peacock',
                         'youtube', 'spotify', 'apple music', 'audible', 'kindle'],
            'software': ['adobe', 'microsoft', 'github', 'jetbrains', 'notion', 'canva',
                        'figma', 'slack', 'zoom', 'dropbox', 'google', 'icloud'],
            'gaming': ['xbox', 'playstation', 'nintendo', 'steam', 'epic', 'discord'],
            'fitness': ['gym', 'fitness', 'peloton', 'strava', 'zwift', 'yoga'],
            'education': ['skillshare', 'masterclass', 'coursera', 'udemy', 'duolingo'],
            'news': ['times', 'wsj', 'economist', 'bloomberg', 'medium', 'substack'],
            'food': ['doordash', 'uber eats', 'grubhub', 'hellofresh', 'blue apron'],
            'cloud': ['aws', 'azure', 'digital ocean', 'heroku', 'netlify', 'vercel']
        }
        
        for category, keywords in categories.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        
        return 'other'


def main():
    """Test the enhanced parser."""
    parser = EnhancedBankParser()
    
    # Test with a file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        results = parser.parse_file(file_path)
        
        if results['success']:
            print(f"\nFound {len(results['subscriptions'])} subscriptions:")
            print(f"Analyzed {results.get('transactions_analyzed', 0)} transactions\n")
            
            for sub in results['subscriptions']:
                print(f"{sub['name']:<30} ${sub['amount']:>6.2f} {sub['frequency']:<10} "
                      f"[{sub['category']}] ({sub['confidence']} confidence)")
            
            print(f"\nTotal Monthly: ${results['total_monthly']:.2f}")
            print(f"Total Annual: ${results['total_annual']:.2f}")
            print(f"Estimated Yearly: ${results['total_monthly'] * 12 + results['total_annual']:.2f}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
    else:
        print("Usage: python bank_parser_enhanced.py <bank_statement.pdf|csv>")


if __name__ == "__main__":
    main()