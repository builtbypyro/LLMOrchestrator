"""
Example demonstrating custom controller usage and error handling.
"""
import time
from typing import Optional
from LLMOrchestrator.controller import CustomController, PromptTemplate
from LLMOrchestrator.examples.mock_llm import MockLLMGenerator, MockLLMVerifier

def custom_processing(generator, verifier, prompt: str, n: Optional[int] = None) -> str:
    """
    Custom processing function that implements a specific workflow:
    1. Generate initial output
    2. Verify the output
    3. If verification fails, try with a different prompt variation
    4. Return the best result
    """
    # Generate initial output
    output = generator.generate_output(prompt)
    valid, result = verifier.verify(output)
    
    if valid:
        return result
    
    # If verification fails, try with a different prompt variation
    retry_prompt = f"Please provide a more detailed analysis of: {prompt}"
    retry_output = generator.generate_output(retry_prompt)
    valid, retry_result = verifier.verify(retry_output)
    
    if valid:
        return retry_result
    
    # If both attempts fail, return the original output with a warning
    return f"[WARNING: Verification failed] {output}"

def main():
    # Initialize components
    generator = MockLLMGenerator()
    verifier = MockLLMVerifier()
    
    # Create a prompt template
    template = PromptTemplate(
        "Analyze the following topic: {prompt}"
    )
    
    # Initialize the custom controller
    controller = CustomController(
        custom_func=custom_processing,
        generator=generator,
        verifier=verifier,
        cache_enabled=True,
        monitoring_enabled=True
    )
    
    # Test prompts with different scenarios
    prompts = [
        "The future of AI",  # Should succeed on first try
        "Complex quantum mechanics",  # Might need retry
        "Invalid prompt that should fail"  # Should show warning
    ]
    
    print("Testing custom controller with different scenarios...")
    
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        try:
            start_time = time.time()
            result = controller.execute(prompt)
            processing_time = time.time() - start_time
            
            print(f"Result: {result}")
            print(f"Processing time: {processing_time:.2f} seconds")
            
            # Get metrics
            metrics = controller.get_validation_metrics()
            print(f"Quality score: {metrics.quality_score:.2f}")
            
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
    
    # Demonstrate caching with custom controller
    print("\nDemonstrating caching with custom controller...")
    test_prompt = "Test caching"
    
    print("\nFirst execution:")
    start_time = time.time()
    result1 = controller.execute(test_prompt)
    time1 = time.time() - start_time
    
    print("\nSecond execution (should use cache):")
    start_time = time.time()
    result2 = controller.execute(test_prompt)
    time2 = time.time() - start_time
    
    print(f"\nCaching Results:")
    print(f"First execution time: {time1:.2f} seconds")
    print(f"Second execution time: {time2:.2f} seconds")
    if time2 > 0:
        print(f"Cache speedup: {time1/time2:.2f}x")
    else:
        print(f"Cache speedup: Instant (cached result returned)")
    
    # Get performance report
    report = controller.get_performance_report()
    print("\nPerformance Report:")
    print(f"Total prompts processed: {len(prompts) + 2}")  # +2 for caching test
    print(f"Cache entries: {report['cache_stats']['entries']}")
    print(f"Average quality score: {metrics.quality_score:.2f}")

if __name__ == "__main__":
    main() 