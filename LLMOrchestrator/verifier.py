import logging
import time
import json

class Verifier:
    """
    Default verifier that checks if generated output is non-empty
    and returns a boolean plus a JSON message with a numeric score.
    Methods:
      - verify(text: str, prompt: str = None) -> (bool, str)
    """
    def __init__(self, custom_verifier=None):
        self.custom_verifier = custom_verifier
        self.logger = logging.getLogger(__name__)

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        try:
            start_time = time.time()
            self.logger.debug(f"Starting verification, text length: {len(text)}")

            # Delegate to custom verifier if provided
            if self.custom_verifier:
                try:
                    valid, result = self.custom_verifier(text, prompt)
                except TypeError:
                    valid, result = self.custom_verifier(text)
                elapsed = time.time() - start_time
                self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, valid={valid}")
                # Custom verifier should return (bool, original_text)
                return valid, result

            # Basic validation: non-empty text passes
            if not text or not text.strip():
                self.logger.warning("Verification failed: Empty output")
                return False, "Empty output"

            # Any non-empty text passes
            elapsed = time.time() - start_time
            self.logger.debug(f"Basic verification completed in {elapsed:.2f}s")
            return True, text

        except Exception as e:
            self.logger.error(f"Error during verification: {str(e)}")
            # On exception, treat as failure
            return False, f"Verification error: {str(e)}"

class CustomVerifier:
    """
    Wrapper for a custom verification function.
    Parameters:
      custom_func: A function that takes output (str) [, prompt: str] and returns (bool, float).
    """
    def __init__(self, custom_func):
        self.custom_func = custom_func
        self.logger = logging.getLogger(__name__)

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        try:
            start_time = time.time()
            self.logger.debug(f"Starting custom verification, text length: {len(text)}")

            try:
                valid, result = self.custom_func(text, prompt)
            except TypeError:
                valid, result = self.custom_func(text)

            elapsed = time.time() - start_time
            self.logger.debug(f"Custom verification completed in {elapsed:.2f}s, valid={valid}")

            # Return the result as-is (should be the original text or modified text)
            return valid, result

        except Exception as e:
            self.logger.error(f"Error in custom verification: {str(e)}")
            return False, f"Verification error: {str(e)}"
