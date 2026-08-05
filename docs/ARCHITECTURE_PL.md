# Architektura i przewodnik programisty

## Układ repozytorium

- `deconv/core/` — niezależne od GUI modele obrazu i PSF, operatory FFT i splotu, metryki, progowanie, poziomy wyświetlania i wspólne procedury numeryczne.
- `deconv/algorithms/` — konkretne implementacje dekonwolucji i rejestr algorytmów.
- `deconv/optim/` — strojenie Auto i izolowany proces obliczeniowy.
- `deconv/denoisers/` — opcjonalne architektury odszumiaczy neuronowych i ich wczytywanie.
- `deconv/gui/` — adaptery GUI, w tym tłumaczone widżety Qt.
- `deconv/api.py` — stabilne punkty wejścia niezależne od Qt do skryptów, notebooków i zadań wsadowych.
- `examples/` — samodzielne programy korzystające z publicznego API.
- `deconv/legacy_runtime.py` — ustabilizowany stan aplikacji Qt i implementacja kart zachowana dla zgodności wstecznej.
- `tests/` — testy regresji numerycznej i procesów.
- `docs/` — dokumentacja dwujęzyczna.

## Internacjonalizacja

Program obsługuje dokładnie języki `en` i `pl`. Angielskie teksty źródłowe są stabilnymi identyfikatorami tłumaczeń w `deconv/i18n.py`. Moduł `deconv/gui/translated_widgets.py` przechowuje angielski tekst źródłowy widżetów, pozycji list, etykiet formularzy, dialogów i dzienników, a następnie tłumaczy je ponownie bez przebudowy programu.

Wartości wewnętrzne są niezależne od języka:

- nazwy algorytmów zapisane w konfiguracji są angielskimi identyfikatorami rejestru;
- listy wyboru pokazują tekst przetłumaczony, lecz logice programu zwracają kanoniczną wartość angielską;
- klucze konfiguracji i komentarze w kodzie pozostają angielskie;
- `language` jest jedynym zależnym od języka polem profilu.

Aby poprawić tłumaczenie, należy edytować `_EXACT_PL` albo tabelę zwrotów komunikatów generowanych w `deconv/i18n.py`. Nie jest potrzebna kompilacja plików `.ts`/`.qm`, ponieważ nie planuje się większej liczby języków.

## Przepływ danych numerycznych

Stan zawiera tablice źródłowe używane przez Reset oraz jedną zatwierdzoną parę obliczeniową:

- obraz wejściowy po progowaniu;
- `calculation_psf` po progowaniu, prostokątnym przycięciu, rzutowaniu na wartości nieujemne i normalizacji sumy do jedności.

Wszystkie algorytmy, inicjalizacja Wienera, inicjalizacja ślepej PSF, metryki ponownego rozmycia i syntetyczne zaburzanie korzystają z tej jawnej PSF obliczeniowej.

## Modele splotu

`deconv/core/operators.py` rozróżnia:

- `linear_same`: liniowy splot z zerowymi warunkami brzegowymi i dokładnym operatorem sprzężonym;
- `circular_fft`: splot kołowy na siatce obrazu, używany przez jawną odwrotność Wienera FFT/IFFT.

Widma stałych PSF są buforowane. Operatory Torch domyślnie używają `float32`.

## Uwagi implementacyjne do blokowej metody Kaczmarza

Plik `deconv/algorithms/kaczmarz.py` implementuje przybliżoną metodę blokową typu ART z użyciem `NumpyLinearSameOperator`. Splot w przód jest obliczany raz na zewnętrzną iterację. Reszty z wybranych bloków obserwacji są sumowane z opcjonalnymi oknami typu podniesiony Hann i normalizowane przez pokrycie blokami. W trybie stabilizowanym wykonywana jest jedna globalna aktualizacja operatorem sprzężonym, dzielona przez energię PSF. Starszy tryb lokalny stosuje lokalne korekty blokowe i normalizuje nakładające się wkłady. Przesuwanie początków bloków, kolejność deterministyczna lub losowa, ograniczenie aktualizacji, tłumienie, nieujemność, TV i odszumianie są niezależnymi ustawieniami. Implementacja celowo nie tworzy macierzy splotu i nie powinna być opisywana jako dokładne rozwiązanie blokowych równań normalnych.


## Publiczne API numeryczne

`deconv.api` jest wspieranym interfejsem integracyjnym poza GUI. Konwertuje tablice NumPy do `GrayImage`/`PSF`, udostępnia kanoniczny rejestr algorytmów, łączy parametry domyślne z parametrami użytkownika, przygotowuje PSF obliczeniową i zwraca natywny `DeconvolutionResult`. Moduł nie importuje Qt.

Stabilne funkcje to `run_deconvolution`, `wiener_filter`, `available_algorithms`, `default_parameters`, `generate_test_image`, `generate_motion_psf`, `disturb_image`, `as_gray_image`, `as_psf` i `save_grayscale`. Stan GUI i `legacy_runtime.py` nie są publicznym API numerycznym.

Szczegóły zawierają `docs/API_PL.md` i `examples/wiener_motion_blur.py`.

## Dodawanie algorytmu

1. Utwórz klasę pochodną od `DeconvolutionAlgorithm` w `deconv/algorithms/`.
2. Zaimplementuj `run()` oraz opcjonalnie metody wsadowego uruchamiania lub oceny.
3. Zarejestruj klasę w `deconv/algorithms/registry.py`.
4. Dodaj w GUI parametry o kanonicznych nazwach angielskich.
5. Dodaj polskie tłumaczenie każdego nowego tekstu widocznego dla użytkownika.
6. Dodaj testy numeryczne i test uruchomieniowy.

## Pakowanie

Metadane projektu i punkty uruchomieniowe są zadeklarowane w `pyproject.toml`. Po instalacji dostępne są polecenia `dekonwolucje` i `dekonwolucje-gui`. PyTorch jest zależnością opcjonalną, ponieważ algorytmy CPU go nie wymagają.

## Dokumentacja PDF

`docs/Deconvolution_GUI_and_Methods_EN.pdf` zawiera zbiorczy angielski opis GUI i metod numerycznych. Kod źródłowy LaTeX jest przechowywany obok pliku PDF.
