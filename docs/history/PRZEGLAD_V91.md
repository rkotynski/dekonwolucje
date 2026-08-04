# Przegląd zmian — wersja v91

## Cel zmiany

Histogramy w karcie 2 zostały umieszczone bezpośrednio nad odpowiadającymi im
suwakami. Położenie na osi poziomej histogramu odpowiada teraz położeniu
suwaka progu.

## Zmiana układu

Dla obrazu pomiarowego oraz PSF zastosowano osobne układy `QGridLayout`:

- histogram znajduje się w wierszu 0, kolumnie 0;
- suwak znajduje się w wierszu 1, kolumnie 0;
- pole liczbowe i przycisk automatycznego progu znajdują się w dalszych
  kolumnach drugiego wiersza.

Histogram i suwak zajmują zatem dokładnie tę samą kolumnę i otrzymują tę samą
szerokość. Kolumna ta jest rozciągliwa, więc zgodność jest zachowywana również
po zmianie rozmiaru okna. Odstęp pionowy między histogramem a suwakiem został
ograniczony do jednego piksela.

## Skala histogramów

Oba histogramy mają:

- 256 przedziałów;
- stały zakres osi poziomej od 0 do 1;
- pionową linię pokazującą aktualnie wybraną wartość progu;
- logarytmiczną skalę liczności;
- oś wykresu rozciągniętą na pełną szerokość płótna.

Histogram PSF nadal przedstawia intensywność znormalizowaną względem maksimum.
Histogram obrazu przedstawia wartości w tej samej skali 0–1, w której działa
suwak `Measured image floor`. Wartości spoza tego zakresu są przycinane tylko
na potrzeby histogramu i nie zmieniają danych obliczeniowych.

## Pozostałe zachowanie

Nie zmieniono sposobu progowania, wyboru części PSF ani działania algorytmów
dekonwolucji. Zmiana dotyczy rozmieszczenia elementów i jednoznacznego
powiązania skali histogramu z położeniem suwaka.

## Testy

- `python -m py_compile deconv/legacy_runtime.py` — bez błędów;
- `PYTHONPATH=. pytest -q` — 12 testów i 15 podtestów algorytmów zakończonych
  powodzeniem.

Środowisko testowe nie zawierało PyQt5, dlatego interfejsu nie uruchomiono,
ale kod GUI przeszedł kontrolę składni.
