from __future__ import annotations

from deconv.core.runtime import clear_image_data_state


def test_clear_image_data_state_preserves_settings_and_removes_data():
    state = {
        "image": object(),
        "degraded": object(),
        "psf": object(),
        "calculation_psf": object(),
        "degradation_psf": object(),
        "result": object(),
        "estimated_psf": object(),
        "last_run_psf": object(),
        "_tab2_threshold_base_degraded": {"data": 1},
        "psf_support_width": 31,
        "psf_support_height": 17,
        "optimized_wiener_k": 0.1,
        "wiener_profile_k": 0.002,
        "calculation_image_shape": (256, 384),
        "zero_padding_enabled": True,
        "algorithm_setting": 123,
        "reference_available": True,
        "reference_source": "loaded",
        "measured_pair_loaded": True,
    }

    clear_image_data_state(state)

    for key in (
        "image", "degraded", "psf", "calculation_psf", "degradation_psf",
        "result", "estimated_psf", "last_run_psf",
        "_tab2_threshold_base_degraded", "psf_support_width",
        "psf_support_height", "optimized_wiener_k",
    ):
        assert key not in state

    assert state["reference_available"] is False
    assert state["reference_source"] is None
    assert state["measured_pair_loaded"] is False
    assert state["wiener_profile_k"] == 0.002
    assert state["calculation_image_shape"] == (256, 384)
    assert state["zero_padding_enabled"] is True
    assert state["algorithm_setting"] == 123
