# LLMOrchestrator

A powerful framework for orchestrating and enhancing Large Language Model (LLM) outputs through multi-model orchestration, chain-of-thought reasoning, and diverse perspectives.

## Overview

LLMOrchestrator is designed to enhance LLM outputs by:
- Orchestrating multiple LLMs in various configurations
- Implementing chain-of-thought reasoning with multiple perspectives
- Leveraging different models for their specific strengths
- Introducing controlled randomness and diversity in outputs
- Providing robust verification and validation mechanisms

## Features

- **Multi-Model Orchestration**: Chain different LLMs in various orders
- **Chain-of-Thought Reasoning**: Break down complex problems into steps
- **Diverse Perspectives**: Combine outputs from different models
- **Adaptive Learning**: Optimize prompts based on performance
- **Parallel Processing**: Handle multiple requests efficiently
- **Caching**: Improve response times for repeated queries
- **Monitoring**: Track performance and quality metrics
- **Custom Verification**: Implement domain-specific validation

## Installation

```bash
pip install llm-orchestrator
```

## Quick Start

```python
from LLMOrchestrator.models import OpenAIModel, LocalModel
from LLMOrchestrator.controller import Controller, PromptTemplate

# Initialize models
generator = OpenAIModel(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)
verifier = LocalModel(
    model_name="facebook/opt-125m",
    device="cpu"
)

# Create a prompt template
template = PromptTemplate(
    "Please provide a detailed analysis of: {prompt}"
)

# Initialize controller
controller = Controller(
    generator=generator,
    verifier=verifier,
    max_iterations=3,
    max_verifications=2,
    parallel_processing=True,
    cache_enabled=True,
    adaptive_learning=True,
    monitoring_enabled=True,
    prompt_template=template
)

# Execute with a prompt
result = controller.execute(
    prompt="Analyze the impact of artificial intelligence on healthcare.",
    stop_early=False
)

# Get performance metrics
metrics = controller.get_validation_metrics()
print(f"Quality Score: {metrics.quality_score}")
print(f"Confidence: {metrics.confidence_score}")
print(f"Processing Time: {metrics.processing_time}s")
```

## Advanced Usage

### Chain-of-Thought Reasoning

```python
# Configure for complex reasoning tasks
controller = Controller(
    generator=generator,
    verifier=verifier,
    max_iterations=5,  # Allow more iterations for complex reasoning
    prompt_template=PromptTemplate(
        "Let's solve this step by step:\n1. First, let's analyze...\n2. Then, we can consider...\n3. Finally, we can conclude...\n\nProblem: {prompt}"
    )
)

result = controller.execute(
    prompt="Explain the relationship between quantum computing and cryptography",
    stop_early=False
)
```

### Parallel Processing with Multiple Models

```python
# Process multiple prompts with different models
prompts = [
    "Generate a technical specification for a new API",
    "Create a user interface design document",
    "Write a security assessment report"
]

results = controller.execute_parallel(
    prompts=prompts,
    max_workers=2,
    stop_early=False
)

# Get performance report
report = controller.get_performance_report()
print(f"Total processing time: {report['total_time']}s")
print(f"Average quality score: {report['avg_quality_score']}")
```

### Custom Controller Implementation

```python
from LLMOrchestrator.controller import CustomController

def custom_processing(output: str) -> str:
    # Add custom processing logic
    return output.upper()

controller = CustomController(
    custom_func=custom_processing,
    generator=generator,
    verifier=verifier,
    parallel_processing=True
)

result = controller.execute(
    prompt="Generate a response",
    n=3  # Generate 3 variations
)
```

## Use Cases

LLMOrchestrator is particularly effective for:

- **Complex Reasoning Tasks**: Break down multi-step problems
- **Content Generation**: Combine different models for writing and editing
- **Technical Documentation**: Generate and validate technical content
- **Research Analysis**: Synthesize information from multiple perspectives
- **Quality Assurance**: Implement multiple verification layers
- **Performance Optimization**: Cache and adapt to improve response times
- **Prompt Engineering**: Optimize prompts through performance tracking

## Configuration

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your-openai-key
```

Configure advanced settings in `orchestration_config.json`:
```json
{
    "max_iterations": 3,
    "max_verifications": 2,
    "parallel_processing": true,
    "cache_enabled": true,
    "adaptive_learning": true,
    "monitoring_enabled": true
}
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{llmorchestrator2024,
  author = {Srinivasan, Trisanth and Patapati, Santosh},
  title = {LLMOrchestrator: A Multi-Model LLM Orchestration Framework for Reducing Bias and Iterative Reasoning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/builtbypyro/LLMOrchestrator}
}
```

## Documentation

Full documentation is available in the [docs](docs/) directory.
