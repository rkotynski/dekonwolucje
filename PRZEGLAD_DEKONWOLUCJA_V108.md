# Deconvolution application v0.108.0 - Clear images

## New action in Tab 1

A bilingual **Clear images / Wyczyść obrazy** button was added next to the existing Reset and Exit actions.

The command removes all data belonging to the current image-processing session:

- loaded or generated disturbed input;
- optional reference image;
- loaded or generated full PSF;
- thresholded/cropped calculation PSF;
- synthetic forward/degradation PSF;
- reconstruction result and estimated blind PSF;
- stored iteration history and cached result metrics;
- Tab 2 threshold-source snapshots and PSF-selection metadata.

The following settings are preserved:

- calculation image width and height;
- visible zero-padding mode;
- PSF generator controls;
- threshold-control values;
- algorithm parameter profiles;
- Auto configuration and CUDA preferences;
- the shared Wiener profile K;
- selected language, settings profile and last image directory.

The action asks for confirmation. If Auto or a reconstruction is running, the application first requests a safe stop. Clearing is postponed if the worker cannot reach a safe stopping point.

## GUI state after clearing

- Tab 1 previews and both histograms are empty.
- Tab 2 previews and histograms are empty and its status reports that no image or PSF is loaded.
- Tab 3 keeps all parameter choices but reports no calculation PSF.
- Tab 4 clears its result, estimated PSF, metrics, progress and iteration history.
- Reference-based metrics remain disabled until a new independent reference is loaded.

## Verification

The complete test suite passed: 55 tests and 15 algorithm subtests. The PDF documentation was rebuilt for version 0.108.0 and visually checked after rendering.
