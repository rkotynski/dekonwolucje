# Przegląd zmian — wersja v96

## 1. Usunięcie ustawień nośnika znanej PSF z karty 1

Z karty **1. Image and PSF** usunięto:

- powiązanie nośnika PSF z rozdzielczością obrazu;
- procent rozmiaru obrazu używany do wyznaczania nośnika;
- przełącznik ograniczania znanej PSF;
- maksymalną szerokość znanej PSF.

Pole **PSF size** pozostało, ale oznacza wyłącznie rozmiar tablicy generowanej
PSF. Nie określa już części używanej w obliczeniach. Dla znanej PSF jedynym
źródłem środka i szerokości wycinka jest karta 2. Ustawienia rozmiaru estymowanej
PSF w metodach ślepych pozostają w karcie 3, ponieważ dotyczą innego problemu.

Starsze profile JSON mogą nadal zawierać usunięte pola. Są one ignorowane.
Wersja schematu ustawień została zwiększona do 96.

## 2. Automatyczna początkowa ramka PSF

Po każdym wczytaniu lub wygenerowaniu nowej PSF program wywołuje
`PSF.automatic_support_selection()`.

Procedura:

1. pobiera piksele z obwodu pełnej tablicy PSF;
2. wyznacza medianę brzegu jako oszacowanie tła;
3. wyznacza odporny rozrzut tła z MAD;
4. odejmuje oszacowane tło;
5. wybiera piksele większe od maksimum dwóch progów:
   - `1e-4` maksimum sygnału ponad tłem,
   - trzech odpornych odchyleń standardowych tła;
6. liczy środek masy wybranego sygnału;
7. tworzy nieparzystą kwadratową ramkę obejmującą aktywne piksele z marginesem
   dwóch pikseli.

Ramka jest propozycją początkową. Nie niszczy pełnej PSF i może być przesuwana
lub skalowana przez użytkownika przed zatwierdzeniem.

Po przejściu do karty 2 program domyślnie przełącza podgląd na **Full PSF array**,
aby proponowana ramka była widoczna.

## 3. Zgodność ramki z operatorem i normalizacja po przycięciu

Wybrany środek jest zapisywany w metadanych PSF także w trybie **Center of
mass**. Dzięki temu `PSF.fitted_to_shape()` używa dokładnie tego samego środka,
który pokazuje czerwona ramka.

Przy przygotowaniu operatora:

1. pełna PSF jest progowana, jeżeli wybrano próg;
2. wybierany jest kwadrat o wskazanym środku i szerokości;
3. brakujące próbki są dopełniane zerami, gdy ramka wychodzi poza tablicę;
4. wybrany wycinek jest ograniczany do wartości nieujemnych;
5. suma wycinka jest normalizowana do 1.

Podgląd **Selected calculation part** pokazuje również jądro znormalizowane do
sumy 1. Normalizacja wyświetlania Matplotlib jest od tego niezależna.

## 4. Wspólna optymalizacja progu PSF i stałej K

W karcie 2 dodano przycisk:

**Optimize PSF floor + Wiener K**

Przycisk jest alternatywą dla **Auto from border**. Optymalizowane są równocześnie:

- próg `PSF floor / peak`;
- stała regularizacji `K` filtru Wienera.

### Dane z niezależną referencją

Dla każdego kandydata tworzona jest rekonstrukcja Wienera. Minimalizowany jest
błąd średniokwadratowy względem obrazu referencyjnego w oryginalnym regionie
obrazu, bez zewnętrznego dopełnienia zerami.

### Dane pomiarowe bez referencji

Nie wolno traktować obrazu rozmytego jako obrazu referencyjnego. W tym przypadku
minimalizowane jest uogólnione kryterium walidacji krzyżowej Wienera (GCV),
wyznaczone w dziedzinie Fouriera.

### Sposób przeszukiwania

- obliczenia odbywają się na podglądzie o dłuższym boku najwyżej 256 pikseli;
- używany jest szeroki logarytmiczny zakres K;
- zakres progów jest budowany wokół odpornego oszacowania tła PSF;
- po etapie zgrubnym wykonywane jest lokalne doprecyzowanie;
- każda kandydatura używa przyciętej PSF znormalizowanej do sumy 1.

Znalezione K jest zapisywane w profilu algorytmu **Wiener**. Jeżeli bieżąca
metoda ma parametr K, pole bieżącej metody również jest aktualizowane. Rozsyłanie
K do wszystkich pozostałych metod pozostaje świadomą operacją wykonywaną
przyciskiem **Copy Wiener settings to all applicable algorithms**.

## 5. Uwagi interpretacyjne

Wspólna optymalizacja progu i K rozwiązuje problem dwuwymiarowy. Te parametry
częściowo kompensują się: zbyt szeroka PSF z dużym tłem może być równoważona
większym K, a agresywne progowanie może prowadzić do mniejszego K. Dlatego
optymalizowanie tylko jednego z nich przy stałej wartości drugiego może dać
mylący wynik.

GCV jest właściwym kryterium bezreferencyjnym dla klasycznego kołowego modelu
Wienera. Nie usuwa ono jednak całkowicie niepewności modelu PSF ani różnicy
między splotem kołowym i liniowym. Wynik automatyczny powinien być traktowany
jako uzasadniona propozycja, którą można porównać z wynikiem progu wyznaczonego
z brzegu PSF.

## 6. Weryfikacja

Wykonano:

- 22 testy jednostkowe i testy uruchomieniowe algorytmów;
- kontrolę zgodności automatycznej ramki z przesuniętą plamką PSF;
- test normalizacji przyciętej i progowanej PSF do sumy 1;
- test zgodności środka zapisanego jako środek masy z operatorem obliczeniowym;
- test wspólnej optymalizacji z referencją;
- test wspólnej optymalizacji GCV bez referencji;
- kompilację składniową wszystkich modułów Pythona.

Środowisko testowe nie zawierało PyQt5, dlatego nie przeprowadzono interaktywnego
testu widżetów. Logika numeryczna, połączenia metod i kod GUI przeszły kontrolę
składni.
