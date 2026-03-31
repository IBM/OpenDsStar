"""Test that ModelBuilder requires cache_dir."""


def test_cache_dir_required():
    """Test that ModelBuilder.build() throws exception without cache_dir."""
    from OpenDsStar.agents.utils.model_builder import ModelBuilder

    try:
        # Try to build without cache_dir - should raise ValueError
        model, model_id = ModelBuilder.build(
            "watsonx/mistralai/mistral-medium-2505", temperature=0.0
        )
        print("❌ FAILED: Should have raised ValueError for missing cache_dir")
        return False
    except ValueError as e:
        if "cache_dir is required" in str(e):
            print(f"✓ Correctly raised ValueError: {e}")
            print("\n✅ Test passed! cache_dir is now required.")
            return True
        else:
            print(f"❌ FAILED: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected exception: {e}")
        return False


if __name__ == "__main__":
    success = test_cache_dir_required()
    exit(0 if success else 1)
