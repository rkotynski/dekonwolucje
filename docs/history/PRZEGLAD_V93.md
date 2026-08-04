# Przegląd zmian — wersja v93

## 1. Zaznaczenie obliczeniowego obszaru PSF w karcie 2

Na podglądzie PSF dodano czerwoną przerywaną ramkę pokazującą kwadratowy obszar,
który zostanie przekazany do operatorów splotu i rekonstrukcji.

Ramka jest wyznaczana na podstawie:

- szerokości **PSF calculation size [px]**;
- aktualnego trybu centrowania: środka masy PSF albo środka geometrycznego;
- tej samej konwencji zaokrąglania do nieparzystej szerokości, której używa
  przygotowanie jądra obliczeniowego.

W trybie **Full PSF array** ramka jest pokazana w układzie współrzędnych pełnej
wczytanej PSF. Jeżeli środek masy znajduje się blisko brzegu i wybrane okno
wychodzi poza tablicę, ramka zachowuje rzeczywiste położenie okna. Część poza
obrazem odpowiada próbkom dopełnianym zerami przez `PSF.centered_window()`.

W trybie **Selected calculation part** ramka otacza cały pokazany wycinek.
Zmiana szerokości, trybu środka albo progu PSF natychmiast odświeża obrys.

## 2. Diagnoza wcześniejszego liczenia kryteriów

W wersji v92 kryteria historii nie były obliczane jednocześnie. Po zakończeniu
algorytmu funkcja `best_iteration_index()` przechodziła po kolejnych klatkach i
dla każdej osobno wywoływała `compute_metrics()`. Wyniki były następnie
zapisywane w pamięci podręcznej, lecz pierwszy postprocessing pozostawał
sekwencyjny.

## 3. Wsadowe obliczanie kryteriów

Dodano funkcję `compute_metrics_batch()`. Bezpośrednio po zakończeniu
rekonstrukcji karta 4 przekazuje do niej całą zapisaną historię.

Wsadowo obliczane są:

- TV i znormalizowane TV;
- PSNR;
- względny błąd ponownego rozmycia;
- względny błąd średniej intensywności;
- kryterium białości reszty;
- łączny koszt bezreferencyjny.

SSIM jest również liczony dla całej partii, za pomocą filtrów
`scipy.ndimage.uniform_filter`, które nie mieszają osi iteracji.

### Torch i CUDA

Główna ścieżka używa Torch w `float32`. Jeżeli algorytm miał włączoną preferencję
CUDA i karta jest dostępna, postprocessing kryteriów odbywa się na GPU.
W przeciwnym razie używana jest wsadowa ścieżka Torch na CPU.

Dla metod ze stałą PSF wszystkie iteracje korzystają ze wspólnego wsadowego FFT.
Dla metod ślepych PSF może być inna w każdej iteracji. Program grupuje wtedy
klatki według rozmiaru jądra i modelu splotu, ale transformaty obrazów i jąder
nadal wykonuje jednocześnie dla całej grupy.

### Zarządzanie pamięcią

Program najpierw próbuje przetworzyć wszystkie iteracje w jednej partii.
Dla bardzo dużych obrazów lub długiej historii automatycznie wyznacza największą
bezpieczną partię na podstawie dostępnej pamięci. W razie błędu pamięci CUDA
próbuje ścieżki CPU, a ostatecznym zabezpieczeniem pozostaje wcześniejsze
obliczanie sekwencyjne.

Dziennik karty 4 podaje:

- urządzenie;
- typ danych;
- liczbę klatek;
- liczbę partii;
- liczbę grup PSF;
- czas postprocessingu;
- informację o ewentualnym przejściu na ścieżkę zapasową.

Po zapełnieniu pamięci podręcznej wybór najlepszej iteracji i przeglądanie
historii nie przeliczają kryteriów ponownie.

## 4. Zgodność numeryczna i wydajność

Dodano testy porównujące wsadowe kryteria z dotychczasowym obliczaniem każdej
klatki osobno. Obejmują one wspólną PSF oraz inną PSF dla każdej iteracji.
Różnice wynikające z użycia `float32` są mniejsze niż przyjęta tolerancja
`3e-5`; typowo wynoszą od około `1e-6` do `1e-8`.

W teście CPU dla 30 klatek 192 × 192:

- ścieżka wsadowa: około 0,050 s;
- wcześniejsza ścieżka sekwencyjna: około 0,198 s;
- przyspieszenie: około 4 razy.

Wynik zależy od rozmiaru obrazu, liczby iteracji, PSF i sprzętu. CUDA powinna
dawać większą korzyść dla dużych historii.

## 5. Testy

Przeszło:

- 15 testów numerycznych, w tym dwa nowe testy kryteriów wsadowych;
- test uruchomieniowy wszystkich 15 zarejestrowanych algorytmów;
- kontrola składni i kompilacja wszystkich modułów.

W środowisku testowym nie było PyQt5, dlatego nie przeprowadzono interaktywnego
testu GUI. Kod interfejsu przeszedł kontrolę składni.
