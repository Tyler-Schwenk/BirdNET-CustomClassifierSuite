"""Test script to verify subset axes work in UI sweep generation."""
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from birdnet_custom_classifier_suite.ui.sweeps.types import SweepState

def test_sweep_state_with_subsets():
    """Test that SweepState correctly handles subset axes."""
    
    # Create state with subset axes
    state = SweepState(
        sweep_name="test_subsets",
        stage_base_yaml="config/stage_1_base.yaml",
        seeds=[123, 456],
        positive_subset_opts=[
            ["curated/bestLowQuality/small"],
            ["curated/bestLowQuality/medium"],
        ],
        negative_subset_opts=[
            ["curated/hardNeg/hardneg_conf_min_85"],
            ["curated/hardNeg/hardneg_conf_min_99"],
        ]
    )
    
    # Get axes dict
    axes = state.get_axes_dict()
    
    print("=== Testing SweepState with Subset Axes ===")
    print(f"\nSweep name: {state.sweep_name}")
    print(f"Seeds: {state.seeds}")
    print(f"Positive subset opts: {state.positive_subset_opts}")
    print(f"Negative subset opts: {state.negative_subset_opts}")
    print(f"\nAxes dict:")
    for key, val in axes.items():
        print(f"  {key}: {val}")
    
    # Verify subset axes are included
    assert "positive_subsets" in axes, "positive_subsets missing from axes dict"
    assert "negative_subsets" in axes, "negative_subsets missing from axes dict"
    assert axes["positive_subsets"] == state.positive_subset_opts
    assert axes["negative_subsets"] == state.negative_subset_opts
    
    # Calculate expected experiment count
    expected_count = (
        len(state.seeds) * 
        len(state.positive_subset_opts) * 
        len(state.negative_subset_opts)
    )
    print(f"\n✓ Expected experiment count: {expected_count} (2 seeds × 2 pos × 2 neg)")
    
    print("\n✓ All tests passed!")
    return True

def test_empty_subsets():
    """Test that empty subsets don't get included in axes."""
    
    state = SweepState(
        sweep_name="test_no_subsets",
        stage_base_yaml="config/stage_1_base.yaml",
        seeds=[123],
        positive_subset_opts=[[]],  # Empty default
        negative_subset_opts=[[]]   # Empty default
    )
    
    axes = state.get_axes_dict()
    
    print("\n=== Testing Empty Subsets ===")
    print(f"Positive subset opts: {state.positive_subset_opts}")
    print(f"Negative subset opts: {state.negative_subset_opts}")
    print(f"Axes dict keys: {list(axes.keys())}")
    
    # Verify empty subsets are NOT included
    assert "positive_subsets" not in axes, "Empty positive_subsets should not be in axes"
    assert "negative_subsets" not in axes, "Empty negative_subsets should not be in axes"
    
    print("✓ Empty subsets correctly excluded from axes")
    return True

if __name__ == "__main__":
    try:
        test_sweep_state_with_subsets()
        test_empty_subsets()
        print("\n=== All UI Integration Tests Passed! ===")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
