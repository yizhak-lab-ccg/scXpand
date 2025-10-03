# Test Organization Guide

This directory contains all tests for the scXpand project, organized according to testing best practices.

## Directory Structure

```
tests/
├── unit/                           # Pure unit tests (isolated, fast)
│   ├── autoencoders/              # Autoencoder model tests
│   │   ├── test_activation_functions.py
│   │   ├── test_ae_losses.py
│   │   ├── test_ae_model_output.py
│   │   ├── test_ae_trainer.py
│   │   ├── test_ae_trainer_early_stopping.py
│   │   ├── test_decoder.py
│   │   ├── test_decoder_transforms.py
│   │   └── test_negative_binomial_parameterization.py
│   ├── core/                      # Core functionality tests
│   │   ├── test_evaluation.py
│   │   └── test_prediction.py
│   ├── data/                      # Data processing unit tests
│   │   ├── test_balanced_batch_sampler.py
│   │   ├── test_data_format.py
│   │   ├── test_data_splitter_basic.py
│   │   ├── test_data_splitter_errors.py
│   │   ├── test_data_util_basic.py
│   │   ├── test_normalization.py
│   │   ├── test_prepare_data.py
│   │   ├── test_preprocess_expression.py
│   │   ├── test_preprocessing_basic.py
│   │   └── test_transforms.py
│   ├── hyperopt/                  # Hyperparameter optimization tests
│   │   ├── test_checkpoint_core.py
│   │   ├── test_checkpoint_resuming.py
│   │   ├── test_error_handling.py
│   │   ├── test_hyperopt.py
│   │   ├── test_linear_hyperopt.py
│   │   ├── test_lr_scheduler_configs.py
│   │   ├── test_optimize.py
│   │   └── test_optimize_integration.py
│   ├── models/                     # Model-specific tests (MLP, Linear, LightGBM)
│   │   ├── test_data_loader.py
│   │   ├── test_inference_preprocessing_compatibility.py
│   │   ├── test_lightgbm_params.py
│   │   ├── test_linear_params.py
│   │   ├── test_linear_trainer.py
│   │   ├── test_mlp_model.py
│   │   ├── test_mlp_trainer.py
│   │   ├── test_model_manager.py
│   │   ├── test_trainer.py
│   │   └── test_zscore_normalization_flag.py
│   └── utils/                      # Utility function tests
│       ├── test_gene_mismatch_handling_all_models.py
│       ├── test_gene_specific_normalization.py
│       ├── test_inference_parameter_propagation.py
│       ├── test_inference_utils.py
│       ├── test_inference_utils_enhanced.py
│       ├── test_io.py
│       ├── test_io_multiprocessing.py
│       ├── test_io_retry_mechanism.py
│       ├── test_metrics.py
│       ├── test_public_api_functions.py
│       ├── test_train_util.py
│       ├── test_training_inference_compatibility.py
│       └── test_training_utils.py
├── integration/                   # Integration tests (medium speed)
│   ├── data_processing/           # Data processing integration tests
│   │   ├── test_dataset.py
│   │   ├── test_dataset_dataformat_integration.py
│   │   ├── test_dataset_gene_comparison.py
│   │   ├── test_dataset_vs_transforms_consistency.py
│   │   ├── test_gene_mismatch_normalization.py
│   │   ├── test_mlp_inference_gene_reordering.py
│   │   ├── test_multiprocessing.py
│   │   ├── test_multiprocessing_dataloaders.py
│   │   ├── test_normalization_consistency.py
│   │   └── test_normalization_integration.py
│   └── pipelines/                # Pipeline integration tests
│       └── test_integration_pipelines.py
├── e2e/                          # End-to-end tests (slow)
│   ├── cli/                      # CLI end-to-end tests
│   │   ├── test_cli_demo.py
│   │   ├── test_main.py
│   │   ├── test_main_cli.py
│   │   ├── test_main_cli_integration.py
│   │   └── test_main_inference.py
│   ├── inference/                # Inference end-to-end tests
│   │   ├── test_pretrained_inference.py
│   │   ├── test_pretrained_model_functions.py
│   │   ├── test_run_inference_edge_cases.py
│   │   ├── test_run_inference_error_handling.py
│   │   └── test_run_inference_integration.py
│   └── training/                 # Training end-to-end tests
│       └── test_training_utils_functions.py
├── fixtures/                     # Shared test fixtures
│   ├── conftest.py              # Main pytest configuration
│   └── conftest_cli.py          # CLI-specific fixtures
└── utils-test/                   # Test utilities and helpers
    └── test_utils.py
```

## Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Single functions, classes, or modules
- **Dependencies**: Minimal mocking, no external services
- **Examples**: Model initialization, data transformation functions, utility functions

### Integration Tests (`integration/`)
- **Purpose**: Test interactions between multiple components
- **Speed**: Medium (1-10 seconds each)
- **Scope**: Component integration, data pipeline flows
- **Dependencies**: Real data processing, but may use mocked external services
- **Examples**: End-to-end data processing, model training workflows

### End-to-End Tests (`e2e/`)
- **Purpose**: Test complete user workflows
- **Speed**: Slow (10+ seconds each)
- **Scope**: Full application workflows
- **Dependencies**: Complete system, real data/external services
- **Examples**: CLI commands, inference pipelines, training workflows

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run by category
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only (slow)
pytest tests/e2e/
```

### Run specific test types
```bash
# Fast tests only
pytest -m "not slow"

# Slow tests only
pytest -m "slow"

# Specific module
pytest tests/unit/autoencoders/
```

### Run by pattern
```bash
# Run all normalization tests
pytest -k "normalization"

# Run all model tests
pytest -k "model"
```

## Test Naming Conventions

- **Unit tests**: `test_<function_or_class_name>.py`
- **Integration tests**: `test_<component1>_<component2>_integration.py` or `test_<workflow>_flow.py`
- **E2E tests**: `test_<user_scenario>_e2e.py`

## Best Practices

1. **Isolation**: Unit tests should not depend on each other
2. **Determinism**: Tests should produce consistent results
3. **Speed**: Prioritize fast feedback loops
4. **Clarity**: Test names should clearly describe what they test
5. **Maintenance**: Keep test code clean and well-documented
6. **Coverage**: Aim for high test coverage on critical paths

## Migration Notes

This structure was reorganized from a flat structure with domain-specific directories. Key changes:

- **Consolidated models**: All model tests (MLP, Linear, LightGBM) moved to `unit/models/`
- **Separated concerns**: Unit vs integration vs E2E tests clearly separated
- **Data processing**: Complex data processing tests moved to integration layer
- **Fixtures centralized**: All test fixtures and conftest files moved to `fixtures/`

## Performance Considerations

- **Unit tests**: Should run < 1 second total
- **Integration tests**: Should run < 1 minute total
- **E2E tests**: Expected to run longer but provide confidence in full workflows

Configure pytest markers to control test execution speed:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Medium speed integration tests
- `@pytest.mark.slow` - Slow end-to-end tests
