from typing import Optional, Callable, Dict, Any, List, Tuple, Set, Union
import concurrent.futures
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from enum import Enum
import threading
from collections import defaultdict
import re

class OutputQuality(Enum):
    EXCELLENT = 4
    GOOD = 3
    FAIR = 2
    POOR = 1

@dataclass
class ValidationMetrics:
    """Enhanced metrics for output validation and quality assessment."""
    confidence_score: float
    processing_time: float
    token_count: int
    refinement_count: int
    validation_checks: List[str]
    quality_score: float = 0.0
    semantic_similarity: float = 0.0
    coherence_score: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class AdaptiveLearning:
    """Manages adaptive learning parameters based on performance."""
    def __init__(self, initial_learning_rate: float = 0.1):
        self.learning_rate = initial_learning_rate
        self.performance_history: List[float] = []
        self.parameter_history: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def update_parameters(self, metrics: ValidationMetrics):
        """Update learning parameters based on performance metrics."""
        with self.lock:
            self.performance_history.append(metrics.quality_score)
            if len(self.performance_history) > 10:
                self.performance_history.pop(0)
            
            # Adjust learning rate based on performance trend
            if len(self.performance_history) >= 2:
                trend = self.performance_history[-1] - self.performance_history[-2]
                self.learning_rate *= (1 + trend * 0.1)
                self.learning_rate = max(0.01, min(1.0, self.learning_rate))

    def get_optimal_parameters(self) -> Dict[str, float]:
        """Get optimized parameters based on learning history."""
        return {
            'learning_rate': self.learning_rate,
            'performance_trend': np.mean(self.performance_history) if self.performance_history else 0.0
        }

class PromptTemplate:
    """Enhanced prompt template management with dynamic adaptation."""
    def __init__(self, base_template: str):
        self.base_template = base_template
        self.variations: Dict[str, str] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_variation(self, name: str, template: str):
        self.variations[name] = template
    
    def get_template(self, variation: str = None) -> str:
        return self.variations.get(variation, self.base_template)
    
    def record_performance(self, variation: str, quality_score: float):
        """Record performance metrics for template variations."""
        with self.lock:
            self.performance_metrics[variation].append(quality_score)
            if len(self.performance_metrics[variation]) > 100:
                self.performance_metrics[variation].pop(0)
    
    def get_best_variation(self) -> str:
        """Get the best performing template variation."""
        if not self.performance_metrics:
            return None
        avg_scores = {
            var: np.mean(scores) 
            for var, scores in self.performance_metrics.items()
        }
        return max(avg_scores.items(), key=lambda x: x[1])[0]

class OutputCache:
    """Enhanced caching system with quality-based retention."""
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 1000, auto_save: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "output_cache.json"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.lock = threading.Lock()
        self.auto_save = auto_save  # Only save automatically if enabled
        self.dirty = False  # Track if cache has unsaved changes
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        with self.lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
                self.dirty = False
            except Exception:
                pass  # Silently fail on cache save errors
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        with self.lock:
            if self._get_cache_size() > self.max_size_bytes:
                self._prune_cache()
            self.cache[key] = value
            self.dirty = True
            if self.auto_save:
                self._save_cache()
    
    def _get_cache_size(self) -> int:
        return sum(len(str(v).encode('utf-8')) for v in self.cache.values())
    
    def _prune_cache(self):
        """Remove low-quality entries to maintain size limit."""
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].get('metrics', {}).get('quality_score', 0),
            reverse=True
        )
        while self._get_cache_size() > self.max_size_bytes and sorted_entries:
            self.cache.pop(sorted_entries.pop()[0])

class Controller:
    """
    Enhanced controller with advanced features for LLM orchestration.
    """
    def __init__(
        self,
        generator,
        verifier,
        max_iterations: int = 3,
        max_verifications: int = 5,
        refinement_generator=None,
        dynamic_iterations=None,
        parallel_processing: bool = False,
        cache_enabled: bool = True,
        retry_attempts: int = 2,
        prompt_template: Optional[PromptTemplate] = None,
        adaptive_learning: bool = True,
        monitoring_enabled: bool = True
    ):
        self.generator = generator
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.max_verifications = max_verifications
        self.refinement_generator = refinement_generator
        self.dynamic_iterations = dynamic_iterations
        self.parallel_processing = parallel_processing
        self.cache_enabled = cache_enabled
        self.retry_attempts = retry_attempts
        self.prompt_template = prompt_template
        self.cache = OutputCache() if cache_enabled else None
        self.metrics = ValidationMetrics(0.0, 0.0, 0, 0, [])
        self.adaptive_learning = AdaptiveLearning() if adaptive_learning else None
        self.monitoring_enabled = monitoring_enabled
        self._setup_logging()

    def _setup_logging(self):
        if self.monitoring_enabled:
            class HTTPFilter(logging.Filter):
                def filter(self, record):
                    return not any(x in record.msg.lower() for x in [
                        'http', 'request', 'response', 'status', 'headers',
                        'urllib3', 'requests', 'openai', 'api'
                    ])

            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            root_logger.handlers = []

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            console_handler.addFilter(HTTPFilter())
            root_logger.addHandler(console_handler)

            file_handler = logging.FileHandler('llm_orchestrator.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)

            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

    def _process_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    raise
                backoff = 2 ** attempt * (1 + np.random.random())
                time.sleep(backoff)

    def _get_cached_output(self, prompt: str) -> Optional[str]:
        if not self.cache_enabled or not self.cache:
            return None
        cached = self.cache.get(prompt)
        if cached and time.time() - cached['timestamp'] < 3600:
            if cached.get('metrics', {}).get('quality_score', 0) >= 0.7:
                return cached['output']
        return None

    def _cache_output(self, prompt: str, output: str):
        if not self.cache_enabled or not self.cache:
            return
        self.cache.set(prompt, {
            'output': output,
            'timestamp': time.time(),
            'metrics': self.metrics.__dict__ if self.metrics else {}
        })

    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a cache key for the given prompt."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()

    def execute(self, prompt: str, **kwargs) -> str:
        """
        Execute the orchestration process for a given prompt.
        
        Args:
            prompt: The input prompt to process
            **kwargs: Additional execution parameters
            
        Returns:
            str: The generated and verified output
        """
        start_time = time.time()
        
        # Check cache first
        cached_output = self._get_cached_output(prompt)
        if cached_output:
            if self.logger:
                self.logger.info(f"Retrieved cached output for prompt: {prompt[:50]}...")
            return cached_output
        
        # Apply prompt template if available
        if self.prompt_template:
            prompt = self.prompt_template.get_template().format(prompt=prompt)
        
        # Determine iterations
        iterations = self.max_iterations
        if self.dynamic_iterations:
            iterations = self.dynamic_iterations(prompt)
        
        # Generate and verify output
        output = None
        refinement_count = 0
        
        for i in range(iterations):
            try:
                if self.logger:
                    self.logger.info(f"Iteration {i+1}/{iterations}")
                
                # Generate output
                output = self._process_with_retry(
                    self.generator.generate_output,
                    prompt
                )
                
                # Verify output
                is_valid, verified_output = self._process_with_retry(
                    self.verifier.verify,
                    output,
                    prompt
                )
                
                if is_valid:
                    output = verified_output
                    break
                    
                refinement_count += 1
                
                # Use refinement generator if available
                if self.refinement_generator:
                    prompt = self.refinement_generator.refine_prompt(prompt, output)
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in iteration {i+1}: {str(e)}")
                if i == iterations - 1:
                    raise
        
        # Update metrics
        processing_time = time.time() - start_time
        token_count = len(output.split()) if output else 0
        
        self.metrics = ValidationMetrics(
            confidence_score=0.8,
            processing_time=processing_time,
            token_count=token_count,
            refinement_count=refinement_count,
            validation_checks=["generated", "verified"],
            quality_score=0.85,
            last_updated=datetime.now()
        )
        
        # Update adaptive learning
        if self.adaptive_learning:
            self.adaptive_learning.update_parameters(self.metrics)
        
        # Cache the output
        self._cache_output(prompt, output)
        
        if self.logger:
            self.logger.info(f"Execution completed in {processing_time:.2f}s")
        
        return output

    def execute_parallel(self, prompts: List[str], max_workers: int = 5) -> List[str]:
        """
        Execute multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to process
            max_workers: Maximum number of parallel workers
            
        Returns:
            List[str]: List of generated outputs
        """
        if not self.parallel_processing:
            return [self.execute(prompt) for prompt in prompts]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.execute, prompt) for prompt in prompts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results

    def get_validation_metrics(self) -> ValidationMetrics:
        """Get the current validation metrics."""
        return self.metrics

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance report.
        
        Returns:
            Dict containing performance metrics and statistics
        """
        cache_entries = len(self.cache.cache) if self.cache else 0
        cache_size_bytes = self.cache._get_cache_size() if self.cache else 0
        
        report = {
            'metrics': self.metrics.__dict__ if self.metrics else {},
            'cache_stats': {
                'enabled': self.cache_enabled,
                'entries': cache_entries,
                'size': cache_size_bytes
            },
            'prompt_template_stats': {}
        }
        
        if self.prompt_template:
            report['prompt_template_stats'] = {
                'variations': len(self.prompt_template.variations),
                'best_variation': self.prompt_template.get_best_variation()
            }
        
        if self.adaptive_learning:
            report['adaptive_learning'] = self.adaptive_learning.get_optimal_parameters()
        
        return report

class CustomController(Controller):
    """Controller that allows custom execution logic through a user-defined function."""
    
    def __init__(self, custom_func, generator=None, verifier=None, **kwargs):
        """
        Initialize CustomController with a custom execution function.
        
        Args:
            custom_func: Callable that takes (generator, verifier, prompt, max_iterations) 
                        and returns the output string
            generator: The generator instance
            verifier: The verifier instance
            **kwargs: Additional arguments passed to parent Controller
        """
        super().__init__(generator=generator, verifier=verifier, **kwargs)
        self.custom_func = custom_func
    
    def execute(self, prompt: str, **kwargs) -> str:
        """
        Execute using the custom function.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional execution parameters
            
        Returns:
            str: The output from custom function
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(prompt)
            cached = self.cache.get(cache_key)
            if cached:
                return cached['output']
        
        # Execute custom function
        result = self.custom_func(
            self.generator, 
            self.verifier, 
            prompt, 
            self.max_iterations
        )
        
        # Cache result if enabled
        if self.cache_enabled:
            cache_data = {
                'output': result,
                'timestamp': time.time(),
                'metrics': self.get_validation_metrics().__dict__ if hasattr(self, 'get_validation_metrics') else {}
            }
            self.cache.set(cache_key, cache_data)
        
        return result