# API Pythona

Algorytmów numerycznych można używać bez PyQt5 i bez uruchamiania GUI. Publiczny interfejs znajduje się w module `deconv.api` i jest także eksportowany z pakietu `deconv`.

## Instalacja

W katalogu repozytorium:

```bash
python -m pip install -e .
```

Opcjonalne algorytmy Torch/CUDA wymagają odpowiedniej instalacji PyTorch:

```bash
python -m pip install -e ".[gpu]"
```

Publiczne API nie importuje Qt. Algorytmy CPU mogą być więc używane w skryptach, notebookach i zadaniach wsadowych bez środowiska graficznego.

## Główne obiekty danych

- `GrayImage` - dwuwymiarowy obraz w skali szarości; tablica znajduje się w `image.data`.
- `PSF` - nieujemna PSF unormowana do sumy jeden; jądro znajduje się w `psf.kernel`.
- `DeconvolutionResult` - wynik zawierający pola `image`, `history`, `metrics` i `info`.

```python
restored_array = result.image.data
iteration_arrays = [frame.data for frame in result.history]
```

W metodach ślepych końcową oszacowaną PSF można odczytać z:

```python
estimated_psf = result.image.metadata.get("estimated_psf")
```

## Funkcje publiczne

### Konwersja i generowanie danych

```python
as_gray_image(array, name="image", normalize=False)
as_psf(array, name="psf")
generate_test_image(width=256, height=None, padding=0)
generate_motion_psf(size=21, angle_deg=35.0)
```

`as_gray_image(..., normalize=False)` zachowuje intensywności i wymaga wartości z zakresu `[0, 1]`. `normalize=True` włącza normalizację min-max. `as_psf()` zeruje wartości ujemne i normalizuje jądro do sumy jeden.

`generate_test_image()` korzysta z tego samego generatora obrazu testowego co GUI. `generate_motion_psf()` tworzy poziomą lub ukośną PSF ruchową.

### Model zaburzania

```python
disturbed = disturb_image(
    image,
    psf,
    noise_sigma=0.01,
    noise_type="Gaussian",
    seed=7,
)
```

Funkcja korzysta z tego samego liniowego splotu `same` z zerowymi warunkami brzegowymi co GUI. Ziarno zapewnia powtarzalność szumu syntetycznego.

### Dostępne algorytmy

```python
names = available_algorithms()
params = default_parameters("Richardson-Lucy")
```

Nazwy algorytmów są kanonicznymi angielskimi identyfikatorami rejestru i nie zależą od języka GUI.

### Uruchamianie dowolnego algorytmu

```python
result = run_deconvolution(
    disturbed,
    psf,
    algorithm="Richardson-Lucy",
    iterations=40,
    epsilon=1e-8,
    non_negative=True,
)
```

Parametry można też przekazać słownikiem:

```python
result = run_deconvolution(
    disturbed,
    psf,
    algorithm="Block Kaczmarz",
    params={
        "iterations": 20,
        "kaczmarz_block_size": 32,
        "kaczmarz_relaxation": 0.15,
    },
)
```

Parametry nazwane mają pierwszeństwo przed słownikiem `params`, a oba źródła mają pierwszeństwo przed wartościami domyślnymi.

### Skrót do filtru Wienera

```python
result = wiener_filter(
    disturbed,
    psf,
    K=2e-3,
    non_negative=True,
)
```

Wywoływana jest ta sama jawna implementacja FFT/IFFT co w GUI. Wynikiem jest część rzeczywista odwrotnej transformaty Fouriera.

### Zapis wyniku

```python
save_grayscale(result.image, "restored.png")
```

Funkcja pomocnicza zapisuje 8-bitowy obraz w skali szarości. Do analiz ilościowych warto zachować również tablicę zmiennoprzecinkową, np. w pliku NumPy lub MAT.

## Kompletny przykład

Repozytorium zawiera plik:

```text
examples/wiener_motion_blur.py
```

Uruchomienie:

```bash
python examples/wiener_motion_blur.py --output-dir wiener_motion_output
```

Program generuje obraz testowy z GUI, ukośną PSF ruchową o rozmiarze 31 pikseli, powtarzalnie zaburzony obraz oraz rekonstrukcję filtrem Wienera. Zapisuje obrazy osobno i wspólne zestawienie `comparison.png`.

## Tablice NumPy

```python
import numpy as np
from deconv.api import run_deconvolution

measured = np.load("measured.npy")
psf = np.load("psf.npy")

result = run_deconvolution(
    measured,
    psf,
    algorithm="Wiener",
    K=1e-3,
)
restored = result.image.data
```

## Modele brzegowe i przygotowanie PSF

API przekazuje algorytmom tę samą PSF obliczeniową: nieujemną i unormowaną do sumy jeden. Jeżeli PSF jest większa od obrazu, zostaje centralnie dopasowana do jego siatki. Zamknięty filtr Wienera używa kołowego modelu FFT, natomiast większość metod iteracyjnych liniowego splotu z zerowymi warunkami brzegowymi i dokładnym operatorem sprzężonym.

## Postęp i zatrzymywanie

Metody iteracyjne przyjmują zaawansowane parametry wykonawcze:

```python
_iteration_callback=current_iteration_callback
_stop_event=threading_or_multiprocessing_event
```

Callback otrzymuje `(current, total)`. Po ustawieniu zdarzenia współpracujące algorytmy zatrzymują się po bieżącej iteracji.

## Stabilność API

Stabilnymi punktami wejścia są funkcje eksportowane z `deconv.api` oraz publiczne pola `GrayImage`, `PSF` i `DeconvolutionResult`. Klasy GUI i obiekty z `legacy_runtime.py` nie należą do publicznego API numerycznego.
