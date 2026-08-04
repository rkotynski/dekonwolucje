# Przegląd zmian w wersji v94

## 1. Automatyczne poziomy obrazu po zakończeniu obliczeń

Po wsadowym obliczeniu kryteriów program nadal wybiera najlepszą zapisaną
iterację według PSNR, GCV albo kryterium bezreferencyjnego. Bezpośrednio po tym
wyborze automatycznie wykonywane jest również **Auto levels** dla wybranej
klatki.

Poziom czerni odpowiada percentylowi 0,5, a poziom bieli percentylowi 99,5.
Operacja korzysta z zapisanego histogramu 4096-binowego i nie uruchamia ponownie
obliczeń kryteriów ani dekonwolucji.

## 2. Interaktywna ramka PSF w karcie 2

W trybie **Full PSF array** czerwoną przerywaną ramkę można edytować myszą:

- przeciągnięcie wnętrza ramki przesuwa jej środek;
- przeciągnięcie krawędzi lub narożnika zmienia rozmiar;
- rozmiar pozostaje kwadratowy i jest sprowadzany do nieparzystej liczby pikseli;
- środek jest ograniczony do współrzędnych wczytanej tablicy PSF;
- sama ramka może częściowo wychodzić poza tablicę, co odpowiada rzeczywistemu
  dopełnianiu brakującej części zerami.

Po ręcznej zmianie program wybiera tryb środka **Manual** i wpisuje współrzędne
`x`, `y` oraz rozmiar do pól liczbowych. Nadal dostępne są również tryby:

- **Center of mass**;
- **Geometric center**;
- **Manual**.

Pola `x` i `y` umożliwiają dodatkowo dokładne wpisanie środka bez użycia myszy.
Przycisk **Apply thresholds / PSF selection now** zapisuje rozmiar, tryb środka
i współrzędne w metadanych PSF używanych przez operator obliczeniowy. Przycisk
**Reset thresholds / PSF selection** przywraca poprzednio zatwierdzony rozmiar,
tryb oraz ręczne współrzędne.

Wspólna metoda `PSF.fitted_to_shape()` obsługuje teraz trzy sposoby centrowania.
Po wycięciu ręcznie wskazany punkt źródłowy jest umieszczany w geometrycznym
środku wynikowego jądra, dlatego ewentualne późniejsze dopasowanie już
przyciętej PSF nie przesuwa jej ponownie.

## 3. Wygodniejsza edycja pól liczbowych

Wszystkie pola typu `QSpinBox` i `QDoubleSpinBox` otrzymały wspólne zachowanie:

- `keyboardTracking=False`: połączone procedury nie są wywoływane po każdym
  znaku wpisywanym w nieukończonej liczbie;
- zmiana zostaje zatwierdzona po naciśnięciu Enter albo przejściu do innego
  elementu interfejsu;
- pola zmiennoprzecinkowe przechowują do 15 miejsc po przecinku niezależnie od
  dawnego limitu prezentacyjnego;
- nieistotne zera na końcu liczby są ukrywane;
- zwiększono minimalną szerokość pól oraz maksymalną długość edytowanego tekstu.

Zmiana usuwa typowy problem, w którym dopisanie cyfry wewnątrz liczby
natychmiast wywoływało procedurę odświeżenia, a ta ponownie formatowała tekst i
przesuwała kursor.

## 4. Kryteria historii iteracji

Pozostawiono wsadowy postprocessing wprowadzony w v93. Wszystkie zapisane
iteracje są oceniane wspólnie przez `compute_metrics_batch()` z użyciem Torch
`float32` i CUDA, gdy jest dostępna. Automatyczne `Auto levels` jest wykonywane
dopiero po zakończeniu tego wsadowego etapu i wyborze najlepszej klatki.

## 5. Weryfikacja

Przeprowadzono:

- kompilację składni wszystkich zmienionych modułów;
- 18 testów numerycznych;
- 15 podtestów uruchomieniowych algorytmów;
- nowy test ręcznego środka PSF i ponownego dopasowania przyciętego jądra.

Wszystkie testy zakończyły się powodzeniem. Środowisko testowe nie zawierało
PyQt5, dlatego interakcji myszą nie uruchomiono w rzeczywistym oknie Qt; kod GUI
przeszedł kontrolę składni i spójności połączeń sygnałów.
