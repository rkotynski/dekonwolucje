# Przegląd zmian — wersja v98

## 1. Pełny zakres poziomów w karcie 4

Suwaki **Black level** i **White level** są teraz zawsze mapowane na pełny zakres
intensywności obrazu, czyli `[0, 1]`. Wcześniej ich końce odpowiadały minimum i
maksimum występującym w aktualnej klatce albo w całej historii, przez co nie
można było wybrać poziomu leżącego poza zakresem rzeczywiście obecnych próbek.

Między poziomem czerni i bieli utrzymywany jest minimalny odstęp równy czterem
krokom suwaka, około `4/4095` pełnego zakresu. Przesuwanie jednego suwaka w
pobliże drugiego automatycznie przesuwa drugi albo zatrzymuje pierwszy tak, aby
zakres wyświetlania nie stał się zerowy. Reguła jest stosowana także przy
wczytywaniu ustawień i automatycznym ustawianiu poziomów.

## 2. Reset ramki do pełnej PSF

W karcie 2 dodano przycisk **Reset frame to full PSF**. Ustawia on:

- szerokość ramki równą pełnej szerokości tablicy PSF;
- wysokość ramki równą pełnej wysokości tablicy PSF;
- środek ramki w geometrycznym środku tablicy;
- podgląd **Full PSF array**.

Znana PSF może obecnie mieć prostokątny obszar obliczeniowy o parzystej lub
nieparzystej szerokości i wysokości. Usunięto automatyczne zmniejszanie
parzystych wymiarów o jeden piksel. Dotyczy to podglądu, przygotowania operatora,
zaburzania i wspólnej optymalizacji progu oraz K. Testy operatora obejmują
jądra parzyste.

## 3. Poprawiona optymalizacja „PSF floor + Wiener K”

Każdy kandydat jest nadal przygotowywany w wymaganej kolejności:

1. odjęcie `floor_fraction * max(PSF)`;
2. wyzerowanie wartości ujemnych;
3. wycięcie dokładnie bieżącej prostokątnej ramki;
4. dopełnienie zerami, jeżeli ramka wychodzi poza tablicę;
5. normalizacja wycinka do sumy 1;
6. utworzenie OTF i wykonanie filtru Wienera.

W v98 rozszerzono zgrubny zakres przeszukiwania K z `1e-10…1e-1` do
`1e-12…1e2`, a lokalne doprecyzowanie może dojść do `1e4`. Przeszukiwanie progu
obejmuje szerszy zakres do `0.95` maksimum PSF. Dla danych bez niezależnej
referencji podstawą pozostaje GCV Wienera, ale do kosztu dodano małą karę za
skorelowaną resztę. Zapobiega to wybieraniu wielu praktycznie równoważnych par
parametrów przez bardzo płaskie GCV.

Optymalizacja rozpoczyna się od K zapisanego w rzeczywistym profilu Wienera, a
nie od niezależnej wartości domyślnej. Po zakończeniu:

- próg jest wpisywany do pola i natychmiast stosowany;
- K jest zapisane w profilu Wienera i, gdy to możliwe, w bieżącym algorytmie;
- bezpośrednio pod przyciskiem pojawia się wynik `przed → po`;
- raportowany jest rozmiar PSF użyty podczas optymalizacji;
- program ponownie przygotowuje faktycznie zastosowaną PSF i pokazuje jej
  rzeczywisty rozmiar oraz sumę po normalizacji.

Jeżeli optymalna para pokrywa się z bieżącą, etykieta pokazuje zerową poprawę,
zamiast sprawiać wrażenie, że przycisk nie zadziałał.

## 4. Karta 1 — kolejność operacji

Przyciski są teraz ułożone w kolejności:

1. **Load image**;
2. **Load PSF**;
3. **Generate test image**;
4. **Generate selected PSF**;
5. **Generate degraded input**.

Usunięto przycisk i funkcję **Load measured image + PSF**. Obraz i PSF można
wczytać niezależnie w dowolnej kolejności.

## 5. Automatyczne uzgadnianie rozmiarów obrazu i PSF

Po każdej operacji wczytania lub generowania obrazu albo PSF program sprawdza,
czy pełna tablica PSF i wszystkie aktywne obrazy mają ten sam rozmiar. Jeżeli
nie, tworzony jest wspólny prostokątny obszar o największej wymaganej szerokości
i wysokości.

Uzgadnianie odbywa się wyłącznie przez centryczne dopełnienie zerami:

- obraz nie jest dodatkowo skalowany;
- PSF nie jest skalowana ani przycinana;
- wybrany środek PSF jest mapowany na środek wspólnej tablicy;
- szerokość i wysokość czerwonej ramki pozostają zachowane;
- metadane obszaru obrazu są przesuwane razem z obrazem.

Przed utworzeniem obrazu zaburzonego wykonywana jest jeszcze końcowa kontrola
zgodności rozmiarów. Sam operator nadal używa wyłącznie fragmentu wskazanego
czerwoną ramką i normalizuje go po przycięciu.

## 6. Testy

Przeszły wszystkie testy numeryczne i uruchomieniowe:

- 27 testów `pytest`, w tym 15 podtestów zarejestrowanych algorytmów;
- test dokładnego zachowania parzystej prostokątnej ramki;
- test normalizacji każdego kandydata optymalizacji;
- test, że wspólna optymalizacja znajduje niegorszą i nietrywialną parę
  parametrów dla danych kontrolnych;
- kontrola składni wszystkich modułów.

Środowisko testowe nie zawierało PyQt5, dlatego interfejsu nie uruchomiono
interaktywnie. Logika numeryczna, połączenia metod i kod GUI przeszły kontrolę
składni.
