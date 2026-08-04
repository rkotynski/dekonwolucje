# Przegląd zmian — wersja v90

## Karta 2

Dodano dwa histogramy o 256 przedziałach:

- histogram obrazu pomiarowego nad regulacją `Measured image floor`;
- histogram PSF nad regulacją `PSF floor / peak`.

Oś liczności jest logarytmiczna. Histogram PSF przedstawia intensywność
podzieloną przez maksimum, dlatego położenie pionowej linii odpowiada dokładnie
wartości suwaka `PSF floor / peak`. Histogramy są wyznaczane z niezmodyfikowanych
danych bazowych bieżącej sesji progowania.

## Wybór części PSF używanej w obliczeniach

Dodano:

- wybór podglądu pełnej tablicy PSF albo części obliczeniowej;
- ustawienie nieparzystej szerokości kwadratowego wycinka w pikselach;
- wybór środka wycinka: środek masy PSF albo środek geometryczny tablicy.

Wybór środka nie jest wyłącznie ustawieniem podglądu. Jest zapisywany w
metadanych PSF i respektowany przez wspólną procedurę przygotowania jądra do
splotu i dekonwolucji. Dotyczy to również algorytmów używających filtru Wienera
jako etapu pomocniczego.

Przycisk `Apply thresholds / PSF selection now` stosuje jednocześnie oba progi,
szerokość wycinka i sposób centrowania. Przycisk
`Reset thresholds / PSF selection` przywraca obraz, PSF, zapisaną PSF pary
pomiarowej, szerokość wycinka i sposób centrowania sprzed rozpoczęcia edycji.

Jeżeli wycinek centrowany geometrycznie nie zawiera żadnej dodatniej wartości
PSF, program zastępuje go impulsem jednostkowym i zapisuje flagę
`empty_selection_replaced_by_impulse` w metadanych. Zapobiega to awarii przy
normalizacji jądra o zerowej sumie.

## Testy

Przeszło 12 testów numerycznych oraz 15 podtestów uruchomieniowych algorytmów.
Dodano test rozróżniający centrowanie środkiem masy i centrowanie geometryczne.
Środowisko testowe nie zawierało PyQt5, dlatego interfejsu nie uruchomiono, ale
moduły zostały sprawdzone przez `py_compile`.
