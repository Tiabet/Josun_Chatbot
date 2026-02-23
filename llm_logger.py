"""
LLM Input/Output Logger
========================
Logs all LLM interactions for debugging purposes.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class LLMLogger:
    """Logger for all LLM inputs and outputs."""
    
    def __init__(self, log_file: str = "llm_interactions.log"):
        self.log_file = log_file
        self.call_counter = 0
        
        # Create Results/Logs directory if it doesn't exist
        os.makedirs("Results/Logs", exist_ok=True)
        
        # Create new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"Results/Logs/llm_log_{timestamp}.txt"
        
        # Initialize log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"LLM Interaction Log - Started at {datetime.now()}\n")
            f.write("=" * 100 + "\n\n")
    
    def log_call(
        self,
        call_type: str,
        input_text: str,
        output_text: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single LLM call.
        
        Args:
            call_type: Type of call (e.g., "Entity Extraction", "Answer Generation")
            input_text: The prompt/input sent to LLM
            output_text: The response from LLM
            context: Additional context (e.g., question, subquestion_id)
        """
        self.call_counter += 1
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"LLM Call #{self.call_counter} - {call_type}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            
            if context:
                f.write("-" * 40 + " CONTEXT " + "-" * 40 + "\n")
                # Pretty print context with specific ordering if possible
                if isinstance(context, dict):
                    for k, v in context.items():
                        f.write(f"[{k.upper()}]:\n{v}\n\n")
                else:
                    f.write(f"{json.dumps(context, ensure_ascii=False, indent=2)}\n")
            
            # Input (Only show if not explicitly suppressed or empty)
            if input_text and input_text.strip() and input_text != "OMITTED":
                f.write("-" * 40 + " FULL PROMPT " + "-" * 40 + "\n")
                f.write(input_text + "\n\n")
            
            # Output
            f.write("-" * 40 + " OUTPUT " + "-" * 40 + "\n")
            f.write(output_text + "\n")
            f.write("=" * 100 + "\n\n")
    
    def log_error(self, call_type: str, error: str, context: Optional[Dict[str, Any]] = None):
        """Log an error during LLM call."""
        self.call_counter += 1
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"LLM Call #{self.call_counter} - {call_type} [ERROR]\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            
            if context:
                f.write(f"Context: {json.dumps(context, ensure_ascii=False, indent=2)}\n")
            
            f.write("=" * 100 + "\n\n")
            f.write(f"ERROR: {error}\n\n")
    
    def finalize(self):
        """Write summary at the end of log."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Log completed at {datetime.now()}\n")
            f.write(f"Total LLM calls: {self.call_counter}\n")
            f.write("=" * 100 + "\n")
        
        print(f"\n✅ LLM interactions logged to: {self.log_file}")
        return self.log_file


# Global logger instance
_global_logger: Optional[LLMLogger] = None

def init_logger(log_file: str = "llm_interactions.log") -> LLMLogger:
    """Initialize global logger."""
    global _global_logger
    _global_logger = LLMLogger(log_file)
    return _global_logger

def get_logger() -> Optional[LLMLogger]:
    """Get global logger instance."""
    return _global_logger

def log_llm_call(call_type: str, input_text: str, output_text: str, context: Optional[Dict] = None):
    """Convenience function to log LLM call."""
    if _global_logger:
        _global_logger.log_call(call_type, input_text, output_text, context)

def log_llm_error(call_type: str, error: str, context: Optional[Dict] = None):
    """Convenience function to log LLM error."""
    if _global_logger:
        _global_logger.log_error(call_type, error, context)

def finalize_log() -> Optional[str]:
    """Finalize and return log file path."""
    if _global_logger:
        return _global_logger.finalize()
    return None
