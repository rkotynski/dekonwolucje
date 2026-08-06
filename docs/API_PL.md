# API Pythona

Wszystkie algorytmy numeryczne oraz wszystkie generatory PSF używane przez GUI są dostępne bez PyQt5 w module `deconv.api`. API przyjmuje obiekty `GrayImage` i `PSF` albo dwuwymiarowe tablice NumPy.

## Zakres API

`available_algorithms()` zwraca komplet 15 metod z rejestru: Wiener, warianty Richardson-Lucy, ślepą Richardson-Lucy, Landwebera, preconditioner Wienera, blokową metodę Kaczmarza, wariant Rosen oraz siedem metod Torch/PyTorch. Każdą można uruchomić przez `run_deconvolution()`, a dla wszystkich dostępne są też funkcje skrótowe.

`available_psf_generators()` zwraca wszystkie rodziny PSF generowane przez GUI:

- `gaussian`;
- `motion` (pozioma lub ukośna zależnie od `angle_deg`);
- `high_frequency`;
- `lens_incoherent`.

## Generowanie danych

```python
from deconv.api import (
    generate_test_image,
    generate_gaussian_psf,
    generate_motion_psf,
    generate_high_frequency_psf,
    generate_lens_incoherent_psf,
    disturb_image,
)

image = generate_test_image(width=384, height=256)
psf = generate_motion_psf(size=31, angle_deg=35.0)
disturbed = disturb_image(image, psf, noise_sigma=0.01, seed=7)
```

Model zaburzania jest zgodny z GUI: liniowy splot `same` z zerowymi warunkami brzegowymi i opcjonalny szum.

## Uruchamianie dowolnej metody

```python
from deconv.api import run_deconvolution

result = run_deconvolution(
    disturbed,
    psf,
    algorithm="Richardson-Lucy",
    iterations=40,
    epsilon=1e-8,
)
restored = result.image.data
history = [frame.data for frame in result.history]
```

`default_parameters(name)` zwraca pełny zestaw parametrów domyślnych wybranej metody.

## Automatyczny dobór parametrów (`Auto`)

API niezależne od GUI udostępnia te same zachowawcze reguły doboru parametrów co GUI:

```python
from deconv.api import AutoTuneOptions, auto_deconvolve

tuning = auto_deconvolve(
    disturbed,
    psf,
    algorithm="Richardson-Lucy",
    reference=image,  # opcjonalny niezależny obraz referencyjny
    params={
        "iterations": 20,
        "epsilon": 1e-8,
        "begin_with_wiener": False,
    },
    auto_options=AutoTuneOptions(
        strategy="quadratic",
        use_torch_equivalent=True,
        tune_boolean=False,
        tune_categorical=False,
    ),
    progress_callback=print,
)

print(tuning.best_params)
restored = tuning.deconvolution_result.image.data
```

`auto_tune_parameters()` jedynie dobiera parametry, chyba że podano `run_best=True`. Funkcja `auto_deconvolve()` zawsze wykonuje końcową rekonstrukcję z zaakceptowanymi parametrami. Obie zwracają `AutoTuningResult`, zawierający parametry początkowe i wybrane, wartości kryterium, liczbę ocenionych kandydatów, czas, komunikat statusu oraz opcjonalny `DeconvolutionResult`.

Przy niezależnym obrazie `reference` Auto maksymalizuje najlepszy PSNR w historii iteracji. Bez referencji zwykły Wiener korzysta z GCV, a pozostałe metody z tego samego kryterium bezreferencyjnego co GUI: zgodności po ponownym rozmyciu, znormalizowanej wariacji całkowitej, zachowania natężenia i białości reszty.

`available_auto_strategies()` zwraca dwa tryby:

- `quadratic` - szybkie przeszukiwanie współrzędnych z lokalnym dopasowaniem paraboli dla parametrów numerycznych;
- `full_batched` - ograniczony lokalny iloczyn kartezjański kandydatów, oceniany wsadowo, gdy implementacja to obsługuje.

Opcje Auto niezależnie sterują doborem parametrów numerycznych, niezależnych parametrów logicznych, kategorycznych, odszumiania, TV oraz K już włączonej inicjalizacji Wienera. Etapy wyłączone w parametrach początkowych pozostają zamrożone razem z ukrytymi parametrami zależnymi. Zwykły algorytm może być strojony szybszym odpowiednikiem Torch batch, a następnie weryfikowany na żądanej implementacji. `use_torch_equivalent=False` wymusza strojenie jednej implementacji.

Opcjonalny `progress_callback` odbiera komunikaty. Obiekt `stop_event` z metodą `is_set()` umożliwia współpracujące przerwanie pomiędzy ocenami kandydatów. Synchroniczne API nie kończy procesu wywołującego podczas długiej pojedynczej iteracji, w odróżnieniu od izolowanego procesu i watchdoga GUI.

Kompletny przykład znajduje się w `examples/auto_richardson_lucy_motion.py`.

## Funkcje skrótowe

CPU: `wiener_filter`, `richardson_lucy_filter`, `richardson_lucy_wiener_filter`, `richardson_lucy_rosen_filter`, `blind_richardson_lucy_filter`, `landweber_filter`, `landweber_wiener_preconditioned_filter`, `block_kaczmarz_filter`.

Torch/PyTorch: `torch_wiener_filter`, `torch_richardson_lucy_filter`, `torch_richardson_lucy_wiener_filter`, `torch_richardson_lucy_rosen_filter`, `torch_landweber_filter`, `torch_adam_tv_map_filter`, `torch_blind_adam_tv_map_filter`.

## Przykłady

Katalog `examples/` zawiera siedem kompletnych programów dla Wienera, Richardson-Lucy, Richardson-Lucy-Wiener, Richardson-Lucy-Rosen, Landwebera i blokowej metody Kaczmarza. Każdy generuje obraz referencyjny, PSF, obraz zaburzony, rekonstrukcję i zestawienie PNG.

```bash
python examples/landweber_motion.py --output-dir landweber_output
python examples/richardson_lucy_rosen_hf.py --output-dir rosen_output
python examples/auto_richardson_lucy_motion.py --output-dir auto_rl_output
```

Metody Wienera używają modelu kołowego FFT, a większość metod iteracyjnych - liniowego splotu z zerowymi warunkami brzegowymi i jego dokładnego operatora sprzężonego.
