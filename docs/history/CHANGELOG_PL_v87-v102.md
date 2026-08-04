# Deconvolution GUI v102



## v102: zamrożone funkcje opcjonalne i bezpieczna optymalizacja progu PSF

- Auto stroi tylko parametry aktywne w chwili rozpoczęcia zadania. Wyłączony Wiener inicjalizacyjny, denoiser, opcjonalny TV i relaksacja Rosena nie mogą zostać chwilowo włączone ani zmienić ukrytych wartości.
- Naprawiono wspólną optymalizację progu PSF i K. GCV wybiera teraz K wyłącznie dla jednej ustalonej PSF; próg bez referencji jest ograniczany odpornymi statystykami tła zmierzonej PSF.
- Automatyczny próg bez referencji nie przekracza 0.25 maksimum PSF, a kandydaci zapadający się do kilku pikseli są odrzucani lub karani.
- Komunikat przycisku pokazuje przedział progu, poziom tła, warunkowe GCV, zachowaną masę oraz efektywną liczbę pikseli PSF.


## v101: wymuszone anulowanie Auto po 5 sekundach

- Obliczenia kandydatów **Auto** i **Auto All** są wykonywane w osobnym, trwałym procesie numerycznym. Proces główny Qt nie uruchamia już długich obliczeń Auto bezpośrednio w wątku roboczym.
- **Cancel Auto** natychmiast ustawia współdzielone żądanie zatrzymania. Algorytmy iteracyjne sprawdzają je po każdej zakończonej iteracji, również w batched Torch/CUDA.
- Jeśli aktualna iteracja, partia FFT/CUDA albo inne pojedyncze wywołanie numeryczne nie zwróci sterowania w ciągu 5 sekund od anulowania, proces pomocniczy jest kończony. Nie jest używane niebezpieczne `QThread.terminate()`, więc główny proces, blokady Qt i stan GUI pozostają nienaruszone.
- Po wymuszonym zatrzymaniu bieżący kandydat nie jest zapisywany, ustawienia Auto nie są częściowo stosowane, a przyciski Auto zostają ponownie odblokowane po zakończeniu wątku sterującego.
- Uruchomienie osobnego procesu może dodać niewielki jednorazowy koszt przy starcie Auto, ale kolejne kandydaty korzystają z tego samego procesu i nie inicjalizują Pythona ani CUDA od początku.


## v100: jawne zatwierdzanie progów i ramki PSF

- Suwaki progów i edycja czerwonej ramki w karcie 2 tworzą wyłącznie ustawienia oczekujące. Obrazy obliczeniowe, histogramy oraz `calculation_psf` zmieniają się dopiero po naciśnięciu **Apply thresholds / PSF selection now**.
- Po zatwierdzeniu pełny podgląd PSF pokazuje tablicę po progowaniu z wartościami równymi zero poza zatwierdzoną ramką. Wybrany prostokąt jest następnie wycinany i normalizowany do sumy 1.
- Czerwoną ramkę można przesuwać, zmieniać jej szerokość i wysokość przez przeciąganie krawędzi oraz zmniejszać lub powiększać kółkiem myszy.
- Usunięto opcję **Use |inverse FFT|**. Filtr Wienera zawsze zwraca część rzeczywistą odwrotnej FFT. Dawny parametr konfiguracyjny jest ignorowany.


## v99: jedna jawna para danych obliczeniowych

- Karty 1 i 2 pokazują bieżący obraz wejściowy oraz jedną `calculation_psf`, czyli PSF po progu, prostokątnym wycięciu i normalizacji do sumy 1. Obok widoczne są histogramy 256-binowe i rzeczywiste rozdzielczości używane przez algorytmy.
- Z karty algorytmów usunięto opcję **Use stored degradation/paired PSF snapshot** oraz alternatywny wybór pełnej PSF dla Wienera. Żaden algorytm nie może już przełączyć się na wcześniejszą kopię jądra.
- W v99 zmiany progów, środka i ramki były stosowane na bieżąco; od v100 wymagają jawnego zatwierdzenia przyciskiem Apply. Dopiero **Reset thresholds / PSF selection** przywraca nieprzetworzone dane wczytane z dysku albo wygenerowane.
- Każdy etap Wienera — samodzielny, inicjalizacyjny i wykonywany między iteracjami — używa wspólnej jawnej implementacji `fft2`/`ifft2` w `float32`; nie jest wywoływana dedykowana funkcja dekonwolucji.
- Obie metody ślepe pobierają wysokość i szerokość estymowanej PSF bezpośrednio z prostokątnej ramki w karcie 2. Ich dawne osobne pola rozmiaru zostały usunięte.



## v98: pełne poziomy, pełna ramka PSF i uzgadnianie rozmiarów

- Suwaki Black/White w karcie 4 obejmują zawsze pełny zakres `[0, 1]` i zachowują minimalny odstęp.
- Karta 2 ma przycisk resetujący ramkę do pełnej szerokości i wysokości PSF; obsługiwane są również parzyste wymiary.
- Optymalizacja progu PSF i K korzysta z rzeczywistego profilu Wienera, szerszego zakresu K i jawnie raportuje zastosowany, przycięty oraz znormalizowany wycinek.
- Przyciski karty 1 mają kolejność: Load image, Load PSF, Generate test image, Generate selected PSF, Generate degraded input. Usunięto wspólne wczytywanie pary.
- Po wczytaniu lub generowaniu obraz i pełna PSF są automatycznie uzgadniane przez centryczne dopełnienie zerami, bez skalowania i przycinania.

## v97: prostokątna ramka PSF i naprawiona optymalizacja progu/K

Karta 2 pozwala wybierać niezależną szerokość i wysokość części PSF. Automatyczna propozycja używa progu 1% maksimum PSF i może mieć kształt prostokątny. Wspólna optymalizacja progu PSF oraz K Wienera wycina dokładnie bieżącą ramkę i normalizuje każdy kandydat do sumy 1. Kryterium GCV korzysta z pełnego widma obrazu, zgodnie z filtrem Wienera.

## v96: wybór nośnika PSF wyłącznie w karcie 2 i wspólna optymalizacja progu z K

- Z karty 1 usunięto wszystkie ustawienia nośnika znanej PSF. Pole **PSF size**
  określa wyłącznie rozmiar generowanej tablicy. Środek i szerokość fragmentu
  obliczeniowego są wybierane wyłącznie w karcie 2.
- Po wczytaniu lub wygenerowaniu PSF program automatycznie proponuje czerwoną
  ramkę obejmującą prawie niezerowy sygnał. Tło i jego rozrzut są oceniane
  odpornie z brzegu PSF, a środek jest środkiem masy sygnału ponad tłem.
  Automatyczny wybór jest tylko punktem startowym i może być ręcznie poprawiony.
- Pełna PSF pozostaje zachowana do podglądu. Przygotowanie operatora pobiera
  wskazany kwadrat, dopełnia go zerami przy wyjściu poza tablicę i zawsze
  normalizuje sumę wybranego jądra do 1.
- W karcie 2 dodano **Optimize PSF floor + Wiener K**. Próg PSF i regularizacja
  Wienera są optymalizowane wspólnie metodą coarse-to-fine na podglądzie do
  256 pikseli. Z niezależną referencją minimalizowany jest MSE rekonstrukcji;
  bez referencji minimalizowane jest GCV Wienera.
- Wybrane K jest zapisywane w profilu Wienera. Można je następnie rozprowadzić
  do pozostałych zgodnych metod istniejącym przyciskiem kopiowania profilu.



## v95: zapamiętywany katalog danych i poprawny pełny podgląd PSF

- Profil ustawień zapisuje `last_image_directory`, czyli katalog ostatniego
  wybranego obrazu lub pliku PSF. Okna **Load image** i **Load PSF** otwierają
  się od tego katalogu także po ponownym uruchomieniu programu.
- Naprawiono zmianę rozmiaru obrazu w `ImageCanvas`. Samo `AxesImage.set_data()`
  nie aktualizuje zasięgu obrazu, dlatego po przełączeniu z małego wycinka na
  **Full PSF array** pełna PSF mogła pozostać narysowana w dawnym małym obszarze
  przy lewym górnym rogu, a reszta płótna była biała.
- Po każdej zmianie tablicy program ustawia obecnie jej pełny zasięg pikselowy.
  Czerwona ramka i obraz korzystają więc dokładnie z tego samego układu
  współrzędnych. Do podglądu PSF zastosowano interpolację `nearest`, aby granice
  pikseli i wybranego okna nie były rozmywane.
- Tryb **Full PSF array** nie przycina PSF. Przycięcie następuje dopiero przy
  przygotowaniu operatora obliczeniowego lub w podglądzie **Selected calculation
  part**. Brakujące próbki okna wychodzącego poza tablicę są nadal dopełniane
  zerami, czyli wizualnie odpowiadają czarnemu otoczeniu.

## v94: automatyczne poziomy, ręczna ramka PSF i elastyczne pola liczbowe

- Po wsadowym obliczeniu kryteriów i wyborze najlepszej iteracji karta 4
  automatycznie wykonuje **Auto levels** dla wybranej klatki.
- W pełnym podglądzie PSF można przesuwać czerwoną ramkę przez przeciągnięcie
  jej wnętrza oraz zmieniać nieparzysty kwadratowy rozmiar przez przeciągnięcie
  krawędzi. Edycja przełącza środek na tryb **Manual** i aktualizuje pola `x`,
  `y` oraz rozmiar.
- Karta 2 oferuje trzy tryby środka: środek masy, środek geometryczny i ręczne
  współrzędne. Apply i Reset obejmują także ręczny środek.
- `PSF.fitted_to_shape()` respektuje ręczne współrzędne zapisane w metadanych,
  a wynikowe przycięte jądro pozostaje stabilne przy ponownym dopasowaniu.
- Wszystkie pola liczbowe zatwierdzają wpis dopiero po Enter lub utracie fokusu.
  Pola zmiennoprzecinkowe dopuszczają 15 miejsc po przecinku, ukrywają zbędne
  zera i mają większą minimalną szerokość.


## v93: obrys obliczeniowej PSF i wsadowe kryteria historii

- W karcie 2 pełny podgląd PSF pokazuje czerwoną przerywaną ramkę dokładnie
  odpowiadającą wybranemu kwadratowemu oknu obliczeniowemu. Położenie ramki
  korzysta z aktualnie wybranego środka masy PSF albo środka geometrycznego.
  Gdy okno wychodzi poza tablicę PSF, ramka zachowuje jego rzeczywiste
  położenie, a brakujące próbki są w operatorze dopełniane zerami.
- W widoku wybranego fragmentu czerwona przerywana ramka otacza cały pokazany
  wycinek.
- Kryteria wszystkich zapisanych iteracji są po zakończeniu rekonstrukcji
  obliczane przez `compute_metrics_batch()`. TV, NTV, PSNR i kryteria
  bezreferencyjne są wektoryzowane w Torch `float32`; reblur jest wykonywany
  wsadowymi FFT, również dla zmieniającej się PSF metod ślepych.
- CUDA jest używana automatycznie, gdy była preferowana dla uruchomionego
  algorytmu i jest dostępna. Bardzo duże historie są dzielone na możliwie duże,
  bezpieczne pamięciowo partie. SSIM jest obliczany dla całej partii za pomocą
  wektoryzowanych filtrów SciPy.
- Dziennik karty 4 podaje urządzenie, liczbę partii, liczbę grup PSF i czas
  postprocessingu. Późniejsze przełączanie iteracji oraz przycisk wyboru
  najlepszej iteracji korzystają już tylko z pamięci podręcznej.


## v92: szybkie poziomy czerni i bieli w karcie 4

- Suwaki w karcie 4 sterują bezpośrednio poziomami intensywności obrazu, a nie
  percentylami obliczanymi ponownie przy każdym ruchu.
- Dla każdej klatki historii tworzony jest jednorazowo histogram 4096-binowy i
  jego dystrybuanta. Etykiety pokazują aktualną wartość intensywności oraz
  odpowiadający jej przybliżony percentyl.
- Zmiana suwaka odświeża wyłącznie obraz rekonstrukcji. Metryki, obraz
  referencyjny, obraz wejściowy i oszacowana PSF nie są ponownie obliczane ani
  rysowane.
- Matplotlib zachowuje istniejący obiekt `AxesImage` i aktualizuje jego dane
  przez `set_data()`, zamiast czyścić osie i tworzyć wykres od początku.
- Podczas przeciągania wyświetlany jest podgląd do 512 pikseli na dłuższym boku,
  a po zwolnieniu suwaka obraz pełnej rozdzielczości.
- **Auto levels** natychmiast ustawia poziomy odpowiadające percentylom 0,5 i
  99,5 z zapisanej dystrybuanty.
- **Optimize display criterion** sprawdza niewielki zestaw kandydatów na obrazie
  zmniejszonym maksymalnie do 192 × 192. Stosuje PSNR, gdy istnieje referencja,
  albo dotychczasowe kryterium bezreferencyjne.
- Metryki każdej zapisanej iteracji są przechowywane w pamięci podręcznej.

## v91: histogramy wyrównane z suwakami progów

- Histogram obrazu pomiarowego znajduje się bezpośrednio nad suwakiem
  **Measured image floor**, a histogram PSF bezpośrednio nad suwakiem
  **PSF floor / peak**.
- Każda para histogram–suwak współdzieli tę samą kolumnę jednego układu
  `QGridLayout`, dlatego oba widżety mają zawsze dokładnie tę samą szerokość,
  również po zmianie rozmiaru okna.
- Oba histogramy mają stałą oś poziomą od 0 do 1 i 256 przedziałów. Oś wykresu
  zajmuje pełną szerokość płótna, a pionowa linia progu odpowiada aktualnej
  pozycji suwaka.
- Usunięto osobne, pełnoszerokościowe wiersze histogramów z formularza karty 2.

## v90: histogramy i wybór obliczeniowej części PSF

- W karcie 2 dodano histogramy obrazu pomiarowego i PSF, wybór widoku pełnej
  PSF lub wycinka obliczeniowego, ustawienie wielkości wycinka oraz centrowanie
  w środku masy PSF albo w środku geometrycznym.
- Przyciski zastosowania i resetowania obejmują progi, rozmiar wycinka i sposób
  jego centrowania.


## v89: ponowny wybór iteracji i automatyczny zakres percentyli

- W karcie 4 dodano przycisk **Select best iteration**. Ponownie wybiera on
  najlepszą klatkę z już zapisanej historii, bez uruchamiania dekonwolucji.
  Stosowane jest dokładnie to samo kryterium co po zakończeniu obliczeń: PSNR
  przy dostępnej niezależnej referencji, GCV dla pomiarowego skanu K Wienera,
  a w pozostałych przypadkach kryterium bezreferencyjne.
- Przycisk **Auto-set percentile range** dobiera położenia suwaków czerni i bieli
  dla aktualnie wyświetlanej klatki. Każda kandydatura jest oceniana po rzeczywistym
  przycięciu do wybranych percentyli i przeskalowaniu do [0, 1]. Kryterium stanowi
  PSNR albo dotychczasowy koszt bezreferencyjny.
- Optymalizacja respektuje tryb **Normalize each iteration independently**. Gdy
  jest wyłączony, oceniany jest wspólny zakres obliczony dla całej historii, więc
  rezultat widoczny po ustawieniu suwaków jest zgodny z wynikiem optymalizacji.
- Przeszukiwanie ma szeroki etap początkowy i trzy etapy doprecyzowania, z końcową
  rozdzielczością 0,1 percentyla, zgodną z rozdzielczością suwaków.


## v88: profil parametrów Wienera (historyczne)

Wersja ta wprowadziła kopiowanie `K` i ustawienia widma mocy szumu między algorytmami. W v100 usunięto alternatywny sposób odczytu IFFT. Dawny wybór źródła PSF został całkowicie usunięty w v99.


## v87: wspólne operatory splotu i diagnostyka zgodności PSF

- Nowy moduł `deconv/core/operators.py` zawiera wspólne, jawnie nazwane modele
  `linear_same` i `circular_fft`, przygotowanie OTF oraz operatory NumPy i
  PyTorch/CUDA. Stała PSF jest transformowana tylko raz i jej widmo jest
  ponownie używane w kolejnych iteracjach.
- Syntetyczne zaburzanie oraz iteracyjne metody nieślepe używają tego samego
  liniowego splotu z zerowymi warunkami brzegowymi. Operator sprzężony jest
  zgodny z dokładnym przycięciem `same`, także dla PSF o parzystych wymiarach.
- Klasyczny filtr Wienera nadal jest odwrotnością w modelu splotu kołowego na
  siatce obrazu. Program zapisuje teraz osobno model fizyczny i model użyty przez
  algorytm oraz raportuje `linear_vs_circular_input_mismatch`.
- W v99 usunięto dawny wybór pomiędzy bieżącą PSF a zapisaną kopią PSF użytej do zaburzania; wszystkie metody korzystają wyłącznie z bieżącej PSF obliczeniowej z karty 2.
- Obliczenia Torch domyślnie używają `float32`; poprawiono również tworzenie
  wektorów parametrów, które wcześniej dla liczb zmiennoprzecinkowych mogło
  niejawnie wybierać `float64`.
- Testy numeryczne znajdują się w katalogu `tests/` i można je uruchomić poleceniem
  `PYTHONPATH=. python -m unittest discover -s tests -v`.

Update: Auto tuning now has a regression guard. Before accepting tuned parameters, the program scores the current parameter set on the validation/Run implementation. Auto-selected parameters are applied only if they do not make that score worse. Otherwise the previous parameters are kept and the Test/Auto log reports the baseline and rejected candidate score.

Run:

```bash
python run_deconvolution_gui.py
```


## v58 update

- Torch-batch variants are no longer shown as separate algorithms in the algorithm list.
- Algorithms with a Torch implementation now show a per-algorithm **PyTorch batch** checkbox.
- The checkbox is enabled by default when an equivalent Torch-batch implementation exists.
- Auto still tunes the Torch-batch implementation for paired algorithms and copies the recovered hyperparameters back to the ordinary/reference implementation.
- The selected method summary now reports whether the run will use `PyTorch batch` or `reference`.

## v59 additions

- Obraz i PSF są obecnie wczytywane osobnymi przyciskami; ich tablice są automatycznie uzgadniane przez centryczne dopełnienie zerami.
- The **Settings** menu can open an existing JSON profile, create a new profile from defaults, save the current profile, or save it under a new name.
- The active settings profile path is shown in the status bar and saved inside the JSON file.


## v60 measured-data handling (historical)

W bieżącej wersji obraz i PSF są wczytywane osobno. Dostępność niezależnego obrazu referencyjnego jest zapisywana w metadanych danych, a metryki PSNR/SSIM są wyłączane, gdy referencji nie ma.


## Source-code architecture (v63)

The numerical implementations no longer live in `legacy_runtime.py`.

- `deconv/core/runtime.py` — image/PSF models, metrics, FFT/Torch helpers and common interfaces.
- `deconv/algorithms/wiener.py` — Wiener and batched Torch Wiener.
- `deconv/algorithms/richardson_lucy.py` — RL, RL-Wiener, RL-Rosen and their Torch batch variants.
- `deconv/algorithms/landweber.py` — Landweber variants.
- `deconv/algorithms/blind.py` — blind RL and blind Adam TV-MAP.
- `deconv/algorithms/adam.py` — Adam TV-MAP.
- `deconv/algorithms/kaczmarz.py` — block Kaczmarz/ART.
- `deconv/algorithms/registry.py` — algorithm registration only.
- `deconv/legacy_runtime.py` — Qt GUI and backward-compatible entry points.

## Original-region metrics and lower thresholding (v64)

Images placed in a zero frame store the exact non-padded content rectangle. PSNR,
SSIM and TV are evaluated only inside that rectangle in the Test tab and during
Auto optimization, including the PyTorch batch and Adam scoring paths. The
artificial zero border therefore does not improve or degrade the reported score.

Tab **2. Degraded input** now provides two lower-threshold controls:

- **Measured image floor [0–1]**: pixels below the selected normalized intensity
  are set to zero.
- **PSF floor / peak [0–1]**: PSF samples below the selected fraction of the PSF
  peak are set to zero; the remaining PSF is normalized to unit sum.

Use **Reset thresholds** to restore the unthresholded image and PSF. For a paired
measured-image + PSF dataset, the thresholded PSF is also used as the paired
calculation PSF. For synthetic data already degraded with an earlier PSF,
regenerate the degraded input after changing the PSF threshold to preserve exact
forward/reconstruction PSF consistency.

## Interactive thresholds, Rosen preset and faster Auto (v65)

Tab **2. Degraded input** now contains sliders synchronized with the numeric
controls for **Measured image floor** and **PSF floor / peak**. Changes are
applied after a short 120 ms pause and the image, PSF and original-region metrics
are refreshed immediately. The unthresholded source arrays are retained, so
moving a slider in either direction does not repeatedly threshold an already
thresholded array.

For **Richardson-Lucy-Rosen**, the button **Set Rosen from Richardson–Lucy
(L=M=1)** copies the common parameters stored for the ordinary Richardson-Lucy
profile, sets both nonlinear exponents to 1 and disables exponent relaxation.
This gives a convenient Rosen baseline configured from the current RL setup.

Auto optimization now offers two strategies:

- **Quadratic coordinate (fast)** (default): samples nearby values for one
  parameter at a time, fits a local concave parabola to numeric scores and tests
  the predicted maximum. Torch-capable candidates are evaluated in batches.
- **Full local grid (thorough)**: retains the previous Cartesian-product search.

The quadratic strategy is usually much cheaper than the full grid, especially
when several numeric parameters are active. It remains derivative-free and is
therefore compatible with integer parameters, checkboxes, denoisers, clipping,
best-iteration selection and other non-smooth operations used by the program.

## Automatic border-based thresholds (v66)

Tab **2. Degraded input** now provides an **Auto from border** button beside
both threshold controls.

- The measured-image floor is set to the mean value of the one-pixel perimeter
  of the original `content_roi`, before the surrounding zero frame is included.
- The PSF floor is computed from the mean value on the perimeter of the preserved
  raw PSF image. Because the GUI threshold is expressed relative to the PSF peak,
  the selected value is `perimeter_mean / raw_peak`.

The selected value is applied immediately and the preview and metrics are
refreshed. The calculation always uses the retained unthresholded source, so the
automatic estimate is not biased by a previously applied threshold.
## Visible zero-padding control (v67)

The first tab contains **Zero-pad image for full convolution**. It is disabled by default.

- Enabled: the image content is reduced and surrounded by a visible zero frame based on the selected PSF support.
- Disabled: the source content fills the configured calculation image.
- If image and PSF array sizes differ, the numerical convolution code still applies the required zero-padding internally. If a supplied PSF array is larger than the image canvas, the image is centered in the smallest compatible canvas without changing its intensity values.

The setting is stored in the selected JSON settings profile and exported to MAT files.


## Image dimensions and padding

The calculation canvas has independent **X (width)** and **Y (height)** settings.
The defaults are **1280 x 1024 pixels**. Visible zero-padding is disabled by
default; convolution routines still perform any internal compatibility padding
required when an image and PSF have different array dimensions. The generated
test image contains broad intrinsic margins around its features.


## Historical note: resolution-linked support (v69; superseded in v96)

Older versions placed known-PSF support controls in Tab 1. In v96 those controls
were removed. Known-PSF centre and width are now selected only in Tab 2. The
resolution-linked width remains an internal initial value only for blind-PSF
working arrays in Tab 3.

## Floor threshold rescaling (v70)

The lower-threshold operations in Tab **2. Degraded input** now subtract the
selected floor and linearly stretch the surviving range back to the original
maximum. For threshold `T` and original maximum `m`:

```text
y = 0                              for x <= T
y = m * (x - T) / (m - T)         for x > T
```

Thus the retained range `[T, m]` becomes `[0, m]` rather than merely being
clipped below `T`. For the PSF, this transformation is followed by the usual
unit-sum normalization required by the deconvolution algorithms.

## Blind PSF constraints and initialization (v71)

Both **Blind Richardson-Lucy** and **PyTorch Blind Adam TV-MAP** expose two
shared PSF options:

- **Constrain estimated PSF to rotational symmetry** projects the initial PSF
  and every subsequent PSF estimate to a centered, non-negative, unit-sum
  rotationally symmetric kernel.
- **Initialize estimated PSF from current known PSF** uses the currently loaded
  or generated PSF as the initial blind estimate, fitted to the selected blind
  support. If this option is disabled, or no PSF is available, a Gaussian PSF
  with the selected size and sigma is used.

The known-PSF initialization option is enabled by default to preserve the prior
Blind Adam behavior and to support experiments where an approximate optical PSF
is available. It may be disabled for fully blind experiments.

## Image and MAT input/output (v74)

The image and PSF loaders preserve native monochrome 8-bit and 16-bit PNG/TIFF
data instead of converting every file to 8-bit grayscale. MATLAB MAT files are
also accepted. When a MAT file contains several numeric two-dimensional arrays,
the GUI asks which variable should be used.

**Save current result** supports 8-bit PNG/TIFF, 16-bit PNG/TIFF, and MAT. The
MAT file stores the reconstruction as a floating-point `result` array together
with the method and parameter summary.

**Save current PSF** supports the same image depths and MAT. For blind methods
it saves the estimated PSF associated with the current iteration when
available. For other methods it saves the exact post-threshold, post-support
PSF used in the most recent deconvolution. The numerical kernel is centered on
a zero-valued canvas with the full spatial resolution used by the latest
deconvolution calculation. Image files are peak-normalized before quantization.
A PSF MAT file contains:

- `psf`: peak-normalized PSF on the full calculation canvas,
- `psf_kernel`: the unit-sum numerical kernel used by the algorithm,
- `psf_kernel_peak_normalized`: the compact kernel normalized by its maximum,
- `calculation_shape_yx`: the full calculation dimensions.

No interpolation or resizing of the saved PSF samples is performed.

## v75 layout update

The first tab now uses a horizontal splitter below the action buttons. Parameter controls are placed in a vertically scrollable panel on the left. The reference image, PSF and degraded-input previews occupy a resizable panel on the right, with the degraded image spanning the lower row.


## v76 layout and Torch shape fixes

- The first tab keeps a wider, scrollable control column and wraps long form rows, preventing the preview panel from clipping controls.
- PyTorch spatial convolution now uses asymmetric zero padding for even-sized PSFs, so `same` convolution always returns exactly the input image dimensions.

## Interruptible Test runs

The **Test** tab runs deconvolution in a Qt worker thread. During iterative
methods it displays the currently completed iteration and the requested total.
The **Stop after current iteration** button requests cooperative cancellation:
the current numerical update is allowed to finish, the last complete image (and
estimated PSF for blind methods) is retained, and no FFT/CUDA operation is
terminated midway.

## v79 reset, PSF redefinition, loaded-size prompts and no-reference quality

- Tab **1. Image and PSF** includes **Reset** and **Exit**. Reset clears all
  loaded images, PSFs, results and histories, restores widget defaults and
  saves the current settings profile. The default calculation grid is now
  **256 x 256**; an existing profile may still override this until Reset or a
  new profile is used.
- Loading an image asks whether the calculation grid should adopt the native
  image size when it differs from the current setting. Atomic loading of a
  measured image and PSF proposes the largest loaded width and height so that
  neither array has to be cropped. Loading a PSF alone asks about resizing only
  when the PSF is larger than the current calculation grid.
- Tab **4. Test** includes **Redefine PSF**. It replaces the current known PSF
  with the PSF corresponding to the displayed reconstruction/iteration. The
  same peak-normalized, full-calculation-resolution representation used by
  **Save current PSF** becomes the visible current PSF, while the compact
  unit-sum kernel remains the numerical PSF for subsequent calculations.
- Without an independent reference, Auto and best-iteration selection no longer
  minimize raw TV alone. They minimize a simple no-reference cost

      relative reblur residual + 0.02 * normalized TV
      + 0.02 * relative mean-intensity error.

  Normalized TV is raw TV divided by the mean absolute image intensity. The
  reblur residual compares the measured image with the reconstruction convolved
  with the known or estimated PSF. All terms are evaluated only in the original
  non-padded image region. This criterion discourages both over-smoothed dark
  images and sharp reconstructions that are inconsistent with the measurement.

## v80 Auto cancellation and Rosen preset fix

- Auto/Auto All busy state is now derived from live `QThread` objects rather than a potentially stale boolean flag.
- Auto controls recover automatically if a thread-finished callback is delayed during Spyder module reloads.
- The Richardson-Lucy-Rosen preset button is disabled while Auto is running or cancellation is still completing.
- Attempting to apply the preset during an active Auto job now shows an explanatory message instead of overwriting the cancellation status.
- The cancellation flag is cleared after cleanup, so later Auto jobs are not blocked by a previous interrupted run.


## Updated numerical defaults (v81)

- Blind deconvolution does not constrain the estimated PSF to rotational symmetry unless explicitly enabled.
- All PyTorch computation paths use float32 by default. Float64 is available as an optional diagnostic precision setting for Torch-capable algorithms.
- Historical v81 behavior: resolution-linked PSF support defaulted to 45% of the smaller image dimension, subject to the former 50% stability limit. This known-PSF policy is superseded by the Tab-2 selection in v96.

## v81 default PSF constraints and Torch precision

- Rotational symmetry is disabled by default for both blind deconvolution methods.
- Float32 is the default numerical format for all Torch paths, including batched algorithms, Adam TV-MAP, and Blind Adam TV-MAP. The shared **Torch numerical precision** checkbox can enable float64 when needed.
- Historical v81 behavior: resolution-linked PSF support defaulted to 45% of the smaller image dimension with a 50% cap. This known-PSF policy is superseded by v96; the rule remains only as an internal blind-PSF initialization heuristic.
- Profiles created by older versions are migrated once so that explicitly stored former defaults do not hide the new v81 defaults.

## v82: logarithmic Wiener-K Auto tuning and Test display scaling

Auto treats **Wiener K** as a positive parameter spanning many orders of
magnitude.  For every algorithm that directly uses a Wiener term, or that has
**Begin with Wiener filter** enabled, Auto now performs:

1. a broad logarithmic scan over the complete K range,
2. a medium log-space refinement around the best decade,
3. a narrow precision refinement around the new best value.

This dedicated K stage is used by both Auto strategies and by hybrid algorithms
that use Wiener filtering for initialization or preconditioning.

Tab **4. Test** now contains display-only normalization controls for stored
iterations:

- **Normalize each iteration independently** gives every frame its own white
  level;
- when disabled, all stored frames share one scale, preserving brightness
  differences between iterations;
- **Maximum percentile** selects the pixel percentile treated as white, from
  60% to 100% (default 97%).

The percentile is measured only inside the original non-padded region.  The
normalization affects visualization only; metrics, saved reconstruction data and
subsequent calculations continue to use the unchanged numerical arrays.

## PSF support policy (v96)

- **Known PSF:** the red-frame centre and width in Tab 2 are authoritative for
  degradation and all non-blind methods.
- **Blind PSF:** Tab 3 retains separate working-width and maximum-width settings
  because the PSF is an optimized variable rather than a fixed known kernel.
- The full loaded PSF is never destroyed by selecting a smaller calculation
  part. Every extracted calculation kernel is normalized to unit sum.

## v84 CUDA and worker-lifecycle safety

This update addresses intermittent native crashes observed after running Auto
and then starting a reconstruction, especially in long-lived Spyder kernels.

- Auto and Test reconstruction now share one serialized numerical-work lock, so
  two CUDA workloads cannot overlap even if GUI state becomes temporarily stale.
- Worker threads synchronize CUDA and release cached allocator/cuFFT resources
  before their QThread is allowed to finish.
- The previous GPU batch benchmark no longer allocates a single probe tensor
  using most of free VRAM. Batch size is estimated conservatively from free
  memory and the algorithm's expected per-candidate footprint.
- If a real Auto batch raises CUDA out-of-memory, it is split recursively into
  smaller batches. A single failing candidate is retried on CPU.
- Auto's worker path no longer reads or writes Qt widgets. Parameter limits and
  execution settings are passed as plain Python values captured before the
  worker starts.
- The application refuses to close or reset while a numerical worker has not
  yet reached a safe stopping point; QThread references are retained until the
  thread truly stops.
- The launcher reuses an existing QApplication when run from an interactive IDE.

For Spyder, it is still preferable to run the GUI in a dedicated console. If
Spyder reports that it reloads `torch`, `torch.ops`, or `torch.classes`, disable
User Module Reloader for PyTorch modules or restart the console before running
the application again. Reloading PyTorch native modules inside the same process
can itself destabilize the kernel independently of the application code.

## v85 PSF centering, display black level, and exact threshold reset

- FFT PSF centering now uses explicit parenthesized integer division. For an
  odd kernel size, the previous expression `-kh // 2` was interpreted by Python
  as `(-kh) // 2`, shifting the kernel one sample too far. The corrected form
  `-(kh // 2)` places the selected PSF center exactly at the Fourier origin.
- Tab **4. Test** adds a **Minimum percentile** slider from 0% to 30%. Samples at
  or below the selected percentile are displayed as black and the surviving
  interval between the selected black and white percentiles is stretched to
  `[0, 1]`. In independent mode each iteration has its own interval; in common
  mode all stored iterations share one black/white interval. This is display
  processing only and never changes saved arrays or metrics.
- **Reset thresholds** now restores complete snapshots of the measured image,
  current PSF, and paired/exact degradation PSF. Threshold metadata and cached
  preview bases are cleared, so a later thresholding session always starts from
  the truly unmodified data rather than from a partially processed object.
- Plain Wiener reconstruction additionally exposes an optional full-array
  circular-FFT PSF compatibility mode and an explicitly nonstandard
  `abs(ifft2(...))` output option for reproducing compact external scripts.

## v86 Wiener GCV and manual logarithmic K scan

When no independent reference image is available, Auto no longer selects the
plain Wiener parameter `K` with the generic image-domain cost. That cost contains
a smoothness term and could prefer an over-regularized result with suppressed
edges. Plain Wiener and Torch batch Wiener now use generalized cross-validation
(GCV) directly in the Fourier domain:

\[
q_K(u,v)=\frac{|H(u,v)|^2}{|H(u,v)|^2+K N(u,v)},
\]

\[
\operatorname{GCV}(K)=
\frac{\operatorname{mean}\left(|(1-q_K)Y|^2\right)/
      \operatorname{mean}(|Y|^2)}
     {\operatorname{mean}(1-q_K)^2}.
\]

The measured image is mean-centered for this calculation so that a large DC
background does not dominate selection. Lower GCV is better. Auto still performs
its broad logarithmic search and local refinements, but candidate comparison for
the plain Wiener filter is now based on GCV rather than normalized TV.

The algorithm panel also provides a manual inspection mode:

- **Generate logarithmic Wiener K scan**;
- **K scan minimum** and **K scan maximum**;
- **K scan points** (default 31).

Each K value produces one stored Test frame. The iteration browser displays the
corresponding K and GCV value, automatically opens the minimum-GCV frame when no
reference is available, and offers **Use displayed K** to copy a visually chosen
value back to the algorithm settings. The scan remains cooperatively interruptible
between K values.

For non-Wiener no-reference selection, the generic cost now gives less weight to
normalized TV and adds a short-range residual-whiteness penalty. This reduces the
systematic preference for excessively smooth images while retaining measurement
consistency and intensity-preservation terms.


## Version 90: Tab 2 histograms and calculation PSF selection

Tab 2 now shows 256-bin logarithmic-count histograms above the measured-image
and PSF floor controls. The PSF histogram uses intensity normalized to the PSF
peak, matching the floor/peak threshold control. Threshold markers move with
the sliders.

The PSF preview can show either the full PSF array or the square part selected
for calculations. The selected part has an odd pixel width and can be centered
at either the thresholded PSF center of mass or the geometric center of the
full PSF image. **Apply thresholds / PSF selection now** commits thresholds,
calculation support and center mode. **Reset thresholds / PSF selection**
restores the complete snapshot from before the editing session.
