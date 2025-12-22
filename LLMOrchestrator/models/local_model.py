"""
Local model implementation using Hugging Face transformers.
"""
import logging
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class LocalModel(BaseModel):
    """Local model implementation using Hugging Face transformers."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """Initialize the local model.
        
        Args:
            model_name: Name of the model to use from Hugging Face
            device: Device to run the model on (cpu/cuda)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
        """
        super().__init__(model_name)  # Add missing parameter
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        logger.info(f"Initializing local model: {model_name}")
        logger.info(f"Using device: {device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Create verification pipeline
        self.verifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=0 if device == "cuda" else -1
        )
        
        logger.info("Local model initialized successfully")
    
    def generate_output(self, prompt: str, **kwargs) -> str:
        """Generate output from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Generate text
            outputs = self.generator(
                prompt,
                max_new_tokens=kwargs.get("max_new_tokens", 50),
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p)
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating output: {e}")
            raise
    
    def validate_output(self, output: str) -> bool:
        """Validate the output.
        
        Args:
            output: Output to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        try:
            # Check for hate speech using RoBERTa
            result = self.verifier(output)[0]
            return result["label"] == "nothate"
            
        except Exception as e:
            logger.error(f"Error validating output: {e}")
            return False
    
    def verify(self, prompt: str, output: str) -> Optional[float]:
        """Verify the quality of the output.
        
        Args:
            prompt: Original prompt
            output: Generated output
            
        Returns:
            Quality score between 0 and 1, or None if verification fails
        """
        try:
            # Check for hate speech
            if not self.validate_output(output):
                return 0.0
            
            # Basic quality checks
            if not output or len(output.split()) < 3:
                return 0.0
            
            # Check if output is too similar to prompt
            if output.lower().startswith(prompt.lower()):
                return 0.0
            
            # Check for repetition
            words = output.lower().split()
            if len(set(words)) / len(words) < 0.5:
                return 0.0
            
            # Calculate quality score based on length and diversity
            length_score = min(len(output.split()) / 20, 1.0)
            diversity_score = len(set(words)) / len(words)
            
            return (length_score + diversity_score) / 2
            
        except Exception as e:
            logger.error(f"Error verifying output: {e}")
            return None
