# Przegląd zmian — wersja v92

## Cel

W karcie 4 usunięto główne źródło opóźnień przy ustawianiu zakresu prezentacji
rekonstrukcji. Wersje v89–v91 ponownie obliczały percentyle z pełnych obrazów i
odświeżały wszystkie wykresy przy każdym kroku suwaka.

## Bezpośrednie poziomy intensywności

Suwaki **Black level** i **White level** mają teraz 4096 pozycji i wskazują
bezpośrednie wartości intensywności w aktywnym zakresie danych. Percentyl nie
jest już współrzędną suwaka. Jest obliczany tylko informacyjnie i wyświetlany
obok wartości, na przykład:

```text
Black: 0.0124  (p1.73)
White: 0.6841  (p99.31)
```

W trybie wspólnym położenia suwaków są mapowane na jeden zakres intensywności
wyznaczony z całej historii. W trybie niezależnym są mapowane na zakres
aktualnie wybranej iteracji.

## Histogram i dystrybuanta w pamięci podręcznej

Dla każdej klatki historii jednorazowo tworzony jest histogram o 4096
przedziałach oraz jego suma skumulowana. Pozwala to:

- uzyskać wartość przybliżonego percentyla bez `np.percentile()`;
- znaleźć poziom odpowiadający zadanemu percentylowi bez ponownego sortowania;
- połączyć statystyki wielu iteracji przez dodanie histogramów, bez łączenia
  pełnych tablic obrazowych.

Histogram jest tworzony w `float32`, a zajmowana przez niego pamięć jest
niezależna od rozmiaru obrazu.

## Szybkie odświeżanie obrazu

Podczas ruchu suwaka:

- nie są ponownie obliczane PSNR, SSIM ani kryteria bezreferencyjne;
- nie jest ponownie rysowana PSF, referencja ani obraz wejściowy;
- nie jest czyszczona oś Matplotlib;
- aktualizowane są tylko dane istniejącego obiektu `AxesImage`;
- zdarzenia są ograniczane timerem 60 ms;
- podczas przeciągania używany jest podgląd o maksymalnym boku 512 pikseli.

Po zwolnieniu suwaka następuje pojedyncze odświeżenie pełnej rozdzielczości.
Metryki dla zapisanych iteracji są dodatkowo przechowywane w pamięci podręcznej.

## Przyciski automatyczne

### Auto levels

Jest to szybka metoda histogramowa. Ustawia:

- poziom czerni na przybliżonym percentylu 0,5;
- poziom bieli na przybliżonym percentylu 99,5.

Nie wykonuje przeszukiwania ani ponownych obliczeń kryterium.

### Optimize display criterion

Jest to dokładniejsza, opcjonalna metoda. Obraz jest najpierw zmniejszany do
maksymalnie 192 × 192 piksele. Następnie sprawdzany jest szeroki zestaw 30 par
poziomów i lokalne doprecyzowanie 3 × 3. Łącznie wykonywanych jest najwyżej
około 40 różnych ocen zamiast około 200 ocen pełnej rozdzielczości.

Kryterium stanowi:

- PSNR po przycięciu i przeskalowaniu, jeśli istnieje niezależna referencja;
- dotychczasowy koszt bezreferencyjny dla danych pomiarowych.

## Zgodność ustawień

Profil ustawień ma schemat 92. Zapisywane są względne położenia suwaków
`display_black_position` i `display_white_position`. Starsze pola percentylowe z
wersji v89–v91 są przy wczytywaniu przybliżenie przeliczane na położenia nowych
suwaków.

## Testy

Przeszło 15 testów numerycznych oraz 15 podtestów uruchomieniowych algorytmów.
Dodano testy:

- zgodności kwantyla i rangi percentylowej wyznaczanych z histogramu;
- łączenia histogramów bez konkatenacji obrazów;
- ograniczonej liczby ocen w szybkim optymalizatorze poziomów.

W środowisku testowym nie było biblioteki PyQt5, dlatego nie wykonano
interaktywnego testu GUI. Kod GUI przeszedł kontrolę składni i kompilację modułów.
