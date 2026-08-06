# Version 0.105.0 - complete Python API and examples

- The public, Qt-independent API now exposes all 15 registered CPU and Torch algorithms through `run_deconvolution()` and dedicated convenience wrappers.
- Added all GUI PSF generators: Gaussian, horizontal/oblique motion, high-frequency, and incoherent-lens PSFs, plus `generate_psf()` and `available_psf_generators()`.
- Reviewed and corrected the contributed API proposal: fixed invalid annotations and syntax, corrected export names and commas, added missing epsilon/normalization forwarding, and aligned wrapper parameter names with the algorithm registry.
- Integrated corrected examples for Richardson-Lucy, Richardson-Lucy-Wiener, Richardson-Lucy-Rosen, Landweber, and Block Kaczmarz alongside the Wiener example.
- Corrected example inconsistencies, including a file named as Gaussian while generating motion blur, an aliased high-frequency PSF, and a nominal motion example using a zero angle.
- Extended English and Polish API documentation and the PDF.
- Updated project authors to Amine Güneş and Rafał Kotyński, University of Warsaw, Faculty of Physics. The PDF remains anonymous as previously requested.
- Version raised to 0.105.0.

Validation: 49 pytest tests and 15 algorithm subtests passed; all six standalone examples completed; the wheel was built and inspected; the 20-page PDF was rendered and visually checked.
