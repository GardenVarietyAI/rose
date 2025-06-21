# Evaluation Scorers

ROSE provides a comprehensive set of scoring functions for model evaluation, following patterns from OpenAI Evals.

## Available Scorers

### exact_match
Compares model output to expected answer after normalization (lowercasing, removing punctuation/articles, normalizing whitespace). Useful for general question-answering tasks where the exact content matters but formatting doesn't.

### f1_score
Calculates token-level F1 score between prediction and ground truth. Provides partial credit based on word overlap, making it useful for longer-form answers where getting some parts right should count.

### fuzzy_match
Uses sequence similarity to allow matches above a configurable threshold (default 80%). Helpful when minor typos or small variations shouldn't count as failures.

### includes
Checks whether the expected answer appears anywhere in the model's output. Good for tasks where the model might provide additional context or explanation around the core answer.

### numeric
Extracts and compares numeric values from text. Supports:
- **Exact matching**: For integer answers or when precision matters
- **Absolute tolerance**: When answers should be within a fixed range (e.g., ±0.01)
- **Relative tolerance**: For percentage-based accuracy (e.g., within 2% of expected)

Handles various numeric formats including decimals, negative numbers, and scientific notation.

## Scorer Selection Guidelines

| Task Type | Recommended Scorer | Why |
|-----------|-------------------|-----|
| Short factual answers | `exact_match` | Handles formatting variations automatically |
| Explanations or essays | `f1_score` | Gives partial credit for relevant content |
| Math problems | `numeric` with tolerance | Handles floating-point precision issues |
| Code output validation | `exact_match` or `includes` | Depends on whether whitespace matters |
| Key information extraction | `includes` | Checks for presence of critical facts |

## Using Scorers in Evaluations

When creating evaluations via the CLI, specify the scorer through the grader type and metric:

- For numeric problems: `--grader numeric_check`
- For fuzzy matching: `--grader text_similarity --metric fuzzy_match`

## Default Scoring Behavior

When running evaluations, ROSE automatically selects appropriate scorers based on the dataset:
- **GSM8K**: Uses numeric matching for mathematical answers
- **HumanEval**: Uses exact match for code generation
- **Custom datasets**: Applies exact_match and f1_score by default

## Attribution

- `exact_match`, `f1_score`: Adapted from the official SQuAD evaluation script (MIT License)
- `fuzzy_match`, `includes`, `numeric`: Inspired by OpenAI Evals implementations (MIT License)
