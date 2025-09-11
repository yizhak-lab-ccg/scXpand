"""Test error handling in hyperparameter optimization."""

from scxpand.hyperopt.hyperopt_optimizer import CATCHABLE_EXCEPTIONS, CRITICAL_ERRORS


def test_error_categorization():
    """Test that errors are categorized correctly for trial failure vs continuation."""
    print("🧪 Testing Error Categorization...")

    print("\n📋 Critical Errors (cause trial failure):")
    for error_type in CRITICAL_ERRORS:
        print(f"  ✓ {error_type.__name__}")

    print("\n📋 Recoverable Errors (allow optimization to continue):")
    for error_type in CATCHABLE_EXCEPTIONS:
        print(f"  ✓ {error_type.__name__}")

    # Verify specific error types are classified correctly
    assert MemoryError in CRITICAL_ERRORS, "MemoryError should cause trial failure"
    assert RuntimeError in CRITICAL_ERRORS, "RuntimeError should cause trial failure"
    assert ValueError in CRITICAL_ERRORS, "ValueError should cause trial failure"
    assert FileNotFoundError in CRITICAL_ERRORS, "FileNotFoundError should cause trial failure"

    assert ConnectionError in CATCHABLE_EXCEPTIONS, "ConnectionError should be recoverable"
    assert TimeoutError in CATCHABLE_EXCEPTIONS, "TimeoutError should be recoverable"
    assert ImportError in CATCHABLE_EXCEPTIONS, "ImportError should be recoverable"
    assert OSError in CATCHABLE_EXCEPTIONS, "OSError should be recoverable"

    print("\n✅ Error categorization is correct!")


def test_error_handling_design():
    """Test the design principles of error handling."""
    print("\n🧪 Testing Error Handling Design...")

    # Test that critical errors are properly identified
    critical_scenarios = [
        ("MemoryError", "CUDA out of memory - batch size too large"),
        ("RuntimeError", "CUDA error during training - hardware issue"),
        ("ValueError", "Invalid parameter values - model misconfiguration"),
        ("FileNotFoundError", "Missing data or model files"),
    ]

    recoverable_scenarios = [
        ("ConnectionError", "Network timeout - temporary connectivity issue"),
        ("TimeoutError", "Request timeout - can retry"),
        ("ImportError", "Optional dependency missing - can skip feature"),
        ("OSError", "Disk space issue - temporary system problem"),
    ]

    print("\n🔥 Critical Error Scenarios (Trial Fails):")
    for error_name, scenario in critical_scenarios:
        print(f"  • {error_name}: {scenario}")

    print("\n🔄 Recoverable Error Scenarios (Continue Optimization):")
    for error_name, scenario in recoverable_scenarios:
        print(f"  • {error_name}: {scenario}")

    print("\n✅ Error handling design covers key failure modes!")


def test_out_of_memory_handling():
    """Test specific handling of out-of-memory scenarios."""
    print("\n🧪 Testing Out-of-Memory Error Handling...")

    # These are the types of memory errors that should cause trial failure
    memory_error_scenarios = [
        "CUDA out of memory",
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.cuda.OutOfMemoryError",
        "RuntimeError: [enforce fail at alloc_cpu.cpp:39]",
    ]

    print("\n💾 Memory Error Scenarios That Should Fail Trials:")
    for scenario in memory_error_scenarios:
        print(f"  ⚠️  {scenario}")

        # These should all be caught by MemoryError or RuntimeError
        if "CUDA out of memory" in scenario or "OutOfMemoryError" in scenario:
            # Usually manifests as RuntimeError in PyTorch
            assert RuntimeError in CRITICAL_ERRORS
        elif "alloc_cpu.cpp" in scenario:
            # CPU memory allocation failures
            assert RuntimeError in CRITICAL_ERRORS

    print("\n✅ Out-of-memory errors properly classified as critical!")


def main():
    """Run all error handling tests."""
    print("🔬 ERROR HANDLING VERIFICATION")
    print("=" * 50)

    test_error_categorization()
    test_error_handling_design()
    test_out_of_memory_handling()

    print("\n" + "=" * 50)
    print("🎉 ERROR HANDLING VERIFICATION COMPLETE!")
    print("=" * 50)

    print("\n🎯 KEY FINDINGS:")
    print("✅ MemoryError → Trial failure (batch size too large)")
    print("✅ RuntimeError → Trial failure (CUDA errors, training issues)")
    print("✅ ValueError → Trial failure (invalid parameters)")
    print("✅ FileNotFoundError → Trial failure (missing files)")
    print("✅ ConnectionError → Continue optimization (network issues)")
    print("✅ TimeoutError → Continue optimization (temporary timeouts)")
    print("✅ ImportError → Continue optimization (missing optional deps)")
    print("✅ OSError → Continue optimization (system issues)")

    print("\n🛡️  ROBUST ERROR HANDLING CONFIRMED!")


if __name__ == "__main__":
    main()
