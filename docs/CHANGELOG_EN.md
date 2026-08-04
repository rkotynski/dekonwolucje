# v103.2 - translation audit and documentation disclosure

- Audited static and dynamic GUI text and completed missing Polish translations.
- Replaced phrase-fragment substitution for dynamic messages with full message-template translation, preventing mixed-language and corrupted words.
- Standardized the Polish GUI term for a degraded input as **obraz zaburzony**.
- Added an LLM-assisted-development disclosure to the project description and documentation.
- Removed author information from the English PDF documentation.

# v103.1 - startup hotfix and PDF documentation

- Added the missing `sys` import required by platform-specific configuration-directory selection.
- Added an English PDF describing the GUI, data flow, convolution models, implemented methods, metrics, Auto tuning, and software architecture.
- Included the PDF in the GitHub source package and installed wheel documentation.

# Changelog

## v103 — bilingual GUI and GitHub-ready project

- Added complete English/Polish GUI switching, persisted in the active JSON settings profile.
- Added translated widget wrappers for form labels, buttons, checkboxes, combo-box displays, tooltips, dialogs, status messages, logs and Matplotlib titles.
- Preserved canonical English algorithm identifiers, configuration keys and source-code comments.
- Added `pyproject.toml`, MIT license, citation metadata, GitHub Actions tests, issue templates and bilingual documentation.
- Moved the default settings file to the user configuration directory and remembered the last selected settings profile.

## v102

- Auto tuning freezes optional stages that were disabled when tuning started.
- Corrected no-reference joint optimization of PSF floor and Wiener K by constraining the floor with robust PSF-background statistics and rejecting collapsed kernels.

## v101

- Added cooperative Auto cancellation with a five-second watchdog and isolated-process termination.

## v100

- Thresholds and the PSF frame become calculation data only after Apply.
- Removed absolute-value inverse-FFT output from Wiener filtering.

## v99 and earlier

Earlier releases introduced one explicit calculation PSF, rectangular PSF selection, batched iteration metrics, fast display levels, consistent FFT/convolution operators and extensive numerical regression tests. Detailed Polish historical notes are retained in `docs/history/`.
