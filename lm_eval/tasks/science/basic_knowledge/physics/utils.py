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
    """Extract answers from Final Answer: \\boxed{} format using Math-Verify"""
    
    # Simplified placeholder filtering for boxed format
    def is_placeholder(text):
        """Check for placeholder content in boxed answers"""
        if not text:
            return True
        text_lower = str(text).lower().strip()
        
        # Key placeholder patterns for boxed format
        placeholders = {
            'your_answer', 
            'your answer', 
            '...', 
            'computed value',
            'your result'
        }
        
        # Check for placeholder patterns
        if any(placeholder in text_lower for placeholder in placeholders):
            return True
            
        # Check for very short responses
        if len(text.strip()) < 1:
            return True
            
        return False
    
    def extract_boxed_answer(text):
        """Extract answer from Final Answer: \\boxed{} format"""
        import re
        
        # Clean up text - remove form feed characters and other issues
        text = text.replace('\x0c', '\\f')  # Replace form feed with \f
        text = text.replace('\x0crac', '\\frac')  # Fix common \frac issue
        
        # Extract complete \boxed{...} format - handle various LaTeX formats
        patterns = [
            r'Final Answer:\s*\\boxed\{',  # Direct \boxed
            r'Final Answer:\s*\\\\boxed\{',  # Double backslash
            r'Final Answer:\s*\\\([^)]*\\boxed\{',  # LaTeX format with \(...\boxed{
            r'\\text\{Final Answer:\s*\}\s*\\boxed\{',  # \text{Final Answer: } \boxed{
            r'Final Answer:\s*\n\s*\\\\?\[\s*.*?\\boxed\{',  # Final Answer: followed by equation block
        ]
        
        match = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                break
        
        if not match:
            # Fallback: look for "Final Answer:" followed by math expression without \boxed{}
            fallback_pattern = r'Final Answer:\s*\n?\s*\\?\[\s*(.*?)\s*\\?\]'
            fallback_match = re.search(fallback_pattern, text, re.IGNORECASE | re.DOTALL)
            if fallback_match:
                # Extract the math content between \[ and \]
                content = fallback_match.group(1).strip()
                return content if content else None
            return None
        
        # Find the matching closing brace
        start_pos = match.end() - 1  # Position of opening brace
        brace_count = 0
        i = start_pos
        
        while i < len(text):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Extract content inside \boxed{...}
                    content = text[start_pos + 1:i].strip()
                    return content if content else None
            i += 1
        
        return None
    
    try:
        from math_verify import parse
        
        filtered_resps = []
        for resp_list in resps:
            filtered = []
            for resp in resp_list:
                # Extract answer from Final Answer: \boxed{} format
                extracted_answer = extract_boxed_answer(resp)
                
                if extracted_answer and not is_placeholder(extracted_answer):
                    # Successfully extracted from boxed format
                    filtered.append(extracted_answer)
                else:
                    # If no "Final Answer:" format found, return empty string
                    filtered.append("")
                    
            filtered_resps.append(filtered)
        return filtered_resps
    
    except ImportError:
        # Fallback when Math-Verify is not available
        filtered_resps = []
        for resp_list in resps:
            filtered = []
            for resp in resp_list:
                # Extract from boxed format even without Math-Verify
                extracted_answer = extract_boxed_answer(resp)
                if extracted_answer and not is_placeholder(extracted_answer):
                    filtered.append(extracted_answer)
                else:
                    # Simple fallback to last non-empty line
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
                # Handle case where pred might be a nested list
                if isinstance(pred, list) and len(pred) > 0:
                    pred = pred[0]  # Take the first element
                elif isinstance(pred, str) and pred.startswith('[') and pred.endswith(']'):
                    # Handle stringified list
                    inner = pred[1:-1].strip()
                    if inner.startswith("'") and inner.endswith("'"):
                        pred = inner[1:-1]
                    elif inner.startswith('"') and inner.endswith('"'):
                        pred = inner[1:-1]
                    else:
                        pred = inner
                
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

def clean_dataset_answer(answer_text):
    """Clean dataset answer text for better Math-Verify compatibility"""
    if not answer_text:
        return answer_text
    
    # Remove common LaTeX formatting that might cause parsing issues
    cleaned = str(answer_text).strip()
    
    # Handle common LaTeX patterns in dataset answers
    # Replace \\ with \ for better parsing
    cleaned = cleaned.replace('\\\\', '\\')
    
    # Handle special LaTeX commands
    cleaned = cleaned.replace('\\mathbb{Z}', 'Z')  # Simplify integer set notation
    cleaned = cleaned.replace('\\times', '*')      # Replace times with multiplication
    
    return cleaned
