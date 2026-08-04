# Przegląd zmian w wersji v89

## Dwa nowe przyciski w karcie 4

### Select best iteration

Przycisk ponownie wykonuje wybór najlepszej pozycji z zapisanej historii
iteracji. Nie uruchamia algorytmu i nie zmienia jego parametrów. Używa tej samej
funkcji `best_iteration_index()`, która jest wywoływana automatycznie po
zakończeniu dekonwolucji.

Kryterium wyboru jest następujące:

- PSNR, gdy dostępny jest niezależny obraz referencyjny;
- minimum GCV dla skanu wartości K filtru Wienera na danych pomiarowych;
- w pozostałych przypadkach maksimum `metric_score`, czyli przede wszystkim
  minimum złożonego kosztu bezreferencyjnego.

### Auto-set percentile range

Przycisk optymalizuje oba suwaki zakresu wyświetlania dla aktualnie wybranej
iteracji. Dla każdej pary percentyli program:

1. oblicza poziom czerni i bieli wewnątrz oryginalnego obszaru obrazu;
2. przycina wartości poniżej i powyżej tych poziomów;
3. skaluje wynik do przedziału [0, 1];
4. ocenia tak przekształcony obraz.

Przy dostępnej referencji maksymalizowany jest PSNR. Dla danych pomiarowych
minimalizowany jest dotychczasowy koszt bezreferencyjny obejmujący zgodność po
ponownym rozmyciu, znormalizowaną wariację całkowitą, zachowanie średniej
intensywności i białość residuum.

Optymalizacja respektuje ustawienie `Normalize each iteration independently`.
W trybie wspólnego zakresu poziomy czerni i bieli są wyznaczane ze wszystkich
zapisanych iteracji, a kryterium jest obliczane dla bieżącej iteracji po użyciu
tego wspólnego zakresu.

Surowe dane rekonstrukcji nie są modyfikowane. Zmieniane są wyłącznie suwaki i
sposób prezentacji wyniku.

## Implementacja i testy

Dodano wspólną funkcję `optimize_percentile_range()` w warstwie numerycznej.
Wykonuje ona szerokie przeszukanie, a następnie trzy etapy doprecyzowania do
rozdzielczości 0,1 percentyla. Wyniki już ocenionych par są buforowane.

Zestaw testów zawiera teraz 11 testów, w tym test zbieżności wyszukiwania
percentyli. Wszystkie testy zakończyły się pomyślnie.
