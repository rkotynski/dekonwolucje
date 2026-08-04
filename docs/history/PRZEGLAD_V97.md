# Przegląd zmian — wersja v97

## 1. Prostokątny obszar PSF

W karcie 2 rozmiar części PSF używanej w obliczeniach jest określany niezależnie przez:

- `Width [px]` — szerokość poziomą;
- `Height [px]` — wysokość pionową.

Oba wymiary są nieparzyste. Czerwoną przerywaną ramkę można nadal przesuwać. Przeciągnięcie lewej lub prawej krawędzi zmienia tylko szerokość, górnej lub dolnej — tylko wysokość, a narożnika — oba wymiary.

Wybrane wymiary i środek są zapisywane w metadanych PSF. Wspólny operator wycina prostokąt przez `PSF.centered_window()`, dopełnia brakujące fragmenty zerami i normalizuje wynik do sumy 1. Starsze profile zawierające tylko `psf_calculation_size` są odczytywane jako wybór kwadratowy.

## 2. Automatyczna propozycja ramki

Próg używany do automatycznego rozpoznawania prawie niezerowego obszaru zwiększono z `1e-4` do `1e-2` maksimum pełnej PSF.

Próg bezwzględny jest większą z wartości:

- `0.01 * max(PSF)`;
- mediany brzegu powiększonej o trzy odporne odchylenia standardowe wyznaczone przez MAD.

Szerokość i wysokość są wyznaczane oddzielnie. Dzięki temu wydłużona PSF, na przykład od ruchu lub aberracji astygmatycznej, nie otrzymuje niepotrzebnie dużej kwadratowej ramki. Jest to tylko propozycja początkowa — ramka nie jest automatycznie zatwierdzana.

## 3. Poprawiona optymalizacja „PSF floor + Wiener K”

Optymalizacja korzysta teraz z jednego jawnego pomocniczego kroku dla każdego kandydata:

1. odjęcie progu `floor_fraction * max(PSF)` i wyzerowanie wartości ujemnych;
2. wycięcie dokładnie aktualnie zaznaczonego prostokąta;
3. dopełnienie zerami, jeżeli ramka wykracza poza pełną tablicę;
4. normalizacja wycinka do sumy 1;
5. utworzenie OTF i ocena filtru Wienera.

Do wyniku diagnostycznego zapisywane są rozmiar wycinka, liczba niezerowych pikseli, suma przed normalizacją i suma po normalizacji.

Dla danych bez obrazu referencyjnego poprawiono kryterium GCV. Wersja v96 usuwała składową stałą obrazu przed oceną. V97 używa pełnego widma, tak jak podstawowa implementacja GCV filtru Wienera. Zwiększa to czułość kryterium na próg i kształt PSF.

Przeszukiwanie uwzględnia bieżący próg i bieżące K jako jednego z kandydatów. Komunikat po obliczeniach pokazuje:

- próg przed i po optymalizacji;
- K przed i po optymalizacji;
- kryterium przed i po optymalizacji;
- rozmiar prostokątnego wycinka;
- sumę PSF po normalizacji.

Wybrane K jest zapisywane w profilu Wienera, a próg jest wpisywany do pola i natychmiast stosowany przez `Apply thresholds / PSF selection now`.

## 4. Zgodność

- Stare ustawienie `psf_calculation_size` jest zachowane jako format zgodności i ustawia oba wymiary.
- Kod wymagający pojedynczej liczby opisującej nośnik otrzymuje większy z dwóch wymiarów, ale nie nadpisuje prostokątnego wyboru.
- Pełna PSF jest nadal zachowywana do podglądu. Przycięcie następuje dopiero w operatorze obliczeniowym lub w widoku `Selected calculation part`.

## 5. Testy

Przeszło 26 testów numerycznych oraz 15 podtestów algorytmów. Dodane testy sprawdzają:

- prostokątną automatyczną ramkę dla wydłużonej PSF;
- próg co najmniej `1e-2` maksimum PSF;
- zachowanie prostokątnego rozmiaru przez `fitted_to_shape()`;
- normalizację prostokątnego wycinka;
- używanie dokładnie wybranego prostokąta przez wspólną optymalizację progu i K;
- brak pogorszenia kryterium względem bieżącej pary parametrów.

Środowisko testowe nie zawierało PyQt5, dlatego interakcja myszą została sprawdzona przez kontrolę logiki i składni, ale nie w uruchomionym oknie GUI.
