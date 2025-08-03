from functools import partial

def process_docs(dataset, task):
    """Filter dataset by specific task type"""
    return dataset.filter(lambda x: x["Task"] == task)

def process_all_docs(dataset):
    """Process all documents without filtering - for datasets with missing Task fields"""
    return dataset

# Process functions for tasks that actually exist in the deep-principle/science_physics dataset
# Based on inspection of exact_match and multiple_choice subsets

# Tasks in both exact_match and multiple_choice subsets:
process_astrophysics_cosmology = partial(process_docs, task="Astrophysics/Cosmology")  # ✅ Available in both subsets
process_quantum_information = partial(process_docs, task="Quantum Information")  # ✅ Available in both subsets
process_condensed_matter_physics = partial(process_docs, task="Condensed Matter Physics") 
process_probability_statistics = partial(process_docs, task="Probability/Statistics")

# Tasks only in multiple_choice subset:
process_computational_physics = partial(process_docs, task="Computational Physics")  # ✅ Available in multiple_choice
process_core_knowledge = partial(process_docs, task="Core Knowledge")  # ✅ Available in multiple_choice
process_high_energy_physics = partial(process_docs, task="High-energy Physics")  # ✅ Available in multiple_choice

def extract_math_answers(resps, docs):
    """Direct Math-Verify answer extraction using native parse() with preprocessing"""
    
    # Minimal placeholder filtering
    def is_placeholder(text):
        """Basic placeholder check"""
        if not text:
            return True
        text_lower = str(text).lower().strip()
        placeholders = {'your mathematical expression', 'your answer', '...'}
        return any(placeholder in text_lower for placeholder in placeholders)
    
    def preprocess_for_math_verify(text):
        """Preprocess text to help Math-Verify's parse() function"""
        import re
        
        # Extract content from <Math>...</Math> tags first
        math_match = re.search(r'<Math>([^<]+)</Math>', text, re.IGNORECASE | re.DOTALL)
        if math_match:
            return math_match.group(1).strip()
            
        # Extract content from <Math>...<\Math> tags (dataset format error)
        math_match = re.search(r'<Math>([^<]+)<\\?/?Math>', text, re.IGNORECASE | re.DOTALL)
        if math_match:
            return math_match.group(1).strip()
        
        # If no Math tags found, try to extract the mathematical expression from the last part
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            # Look at the last few lines for the final answer
            last_lines = lines[-3:] if len(lines) >= 3 else lines
            
            for line in reversed(last_lines):  # Start from the last line
                # Look for patterns like "c ∼ expression" or "answer = expression"
                math_expr_patterns = [
                    r'.*[∼~=]\s*(.+)$',  # c ∼ d^NC or answer = expression
                    r'.*:\s*(.+)$',      # Therefore: expression
                    r'^\s*\\?\[\s*(.+)\s*\\?\]$',  # \\[expression\\]
                    r'^\s*\$\$(.+)\$\$\s*$',       # $$expression$$
                ]
                
                for pattern in math_expr_patterns:
                    match = re.search(pattern, line)
                    if match:
                        extracted = match.group(1).strip()
                        if extracted and len(extracted) > 1 and 'd' in extracted:  # Likely final answer with 'd'
                            return extracted
            
            # If no specific pattern found, try last line with mathematical symbols
            for line in reversed(last_lines):
                if any(symbol in line for symbol in ['d^', '^', '∼', '~', '=']):
                    # Try to extract the expression part after common words
                    expr_after_patterns = [
                        r'.*(?:answer|final|result|thus|therefore|hence)[^:]*:\s*(.+)$',
                        r'.*[∼~=]\s*(.+)$',
                        r'^(.+)$'  # Last resort
                    ]
                    
                    for pattern in expr_after_patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            extracted = match.group(1).strip()
                            if extracted and len(extracted) > 1:
                                return extracted
            
            # Final fallback: return the last line
            return lines[-1]
        
        # Look for mathematical expressions in LaTeX format (fallback)
        latex_patterns = [
            r'\\?\[[^\]]+\\?\]',  # [expression]
            r'\$\$[^$]+\$\$',     # $$expression$$
            r'\$[^$]+\$',         # $expression$
        ]
        
        # Find all matches and take the last one
        all_matches = []
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            all_matches.extend(matches)
        
        if all_matches:
            return all_matches[-1].strip()
        
        # Return original text for Math-Verify to handle other formats
        return text
    
    try:
        from math_verify import parse
        
        filtered_resps = []
        for resp_list in resps:
            filtered = []
            for resp in resp_list:
                try:
                    # Preprocess the response to help Math-Verify
                    preprocessed = preprocess_for_math_verify(resp)
                    
                    # Use Math-Verify's parse() function to extract expressions
                    parsed_results = parse(preprocessed)
                    
                    if parsed_results:
                        # Take the first successfully parsed expression (SymPy object)
                        extracted = str(parsed_results[0])
                        if not is_placeholder(extracted):
                            filtered.append(extracted)
                        else:
                            # Try the string version if available
                            if len(parsed_results) > 1:
                                extracted = parsed_results[1]
                                if not is_placeholder(extracted):
                                    filtered.append(extracted)
                                else:
                                    # Use preprocessed result
                                    if not is_placeholder(preprocessed):
                                        filtered.append(preprocessed)
                                    else:
                                        # Simple fallback: last non-empty line
                                        lines = [line.strip() for line in resp.strip().split('\n') if line.strip()]
                                        filtered.append(lines[-1] if lines else resp.strip())
                            else:
                                # Use preprocessed result
                                if not is_placeholder(preprocessed):
                                    filtered.append(preprocessed)
                                else:
                                    # Simple fallback: last non-empty line
                                    lines = [line.strip() for line in resp.strip().split('\n') if line.strip()]
                                    filtered.append(lines[-1] if lines else resp.strip())
                    else:
                        # Math-Verify parse failed, use preprocessed result
                        if not is_placeholder(preprocessed):
                            filtered.append(preprocessed)
                        else:
                            # Simple fallback: last non-empty line
                            lines = [line.strip() for line in resp.strip().split('\n') if line.strip()]
                            filtered.append(lines[-1] if lines else resp.strip())
                except Exception:
                    # Basic fallback with preprocessing
                    preprocessed = preprocess_for_math_verify(resp)
                    if not is_placeholder(preprocessed):
                        filtered.append(preprocessed)
                    else:
                        lines = [line.strip() for line in resp.strip().split('\n') if line.strip()]
                        filtered.append(lines[-1] if lines else resp.strip())
                    
            filtered_resps.append(filtered)
        return filtered_resps
    
    except ImportError:
        # Simple fallback when Math-Verify is not available
        filtered_resps = []
        for resp_list in resps:
            filtered = []
            for resp in resp_list:
                lines = [line.strip() for line in resp.strip().split('\n') if line.strip()]
                filtered.append(lines[-1] if lines else resp.strip())
            filtered_resps.append(filtered)
        return filtered_resps

def math_verify_score(predictions, references):
    """Direct Math-Verify verification using native parse() + verify()"""
    try:
        from math_verify import parse, verify
        
        correct = 0
        for pred, ref in zip(predictions, references):
            try:
                # Parse both prediction and reference using Math-Verify
                parsed_pred = parse(str(pred))
                parsed_ref = parse(str(ref))
                
                # Use parsed expressions if available, otherwise use strings
                pred_expr = parsed_pred[0] if parsed_pred else str(pred)
                ref_expr = parsed_ref[0] if parsed_ref else str(ref)
                
                # Use Math-Verify's verify() function for comparison
                if verify(ref_expr, pred_expr):  # Note: gold first, pred second as per docs
                    correct += 1
                    
            except Exception:
                # Simple string fallback
                if str(pred).strip().lower() == str(ref).strip().lower():
                    correct += 1
        
        return correct / len(predictions) if predictions else 0
    
    except ImportError:
        # Basic exact match fallback
        correct = 0
        for pred, ref in zip(predictions, references):
            if str(pred).strip().lower() == str(ref).strip().lower():
                correct += 1
        return correct / len(predictions) if predictions else 0
