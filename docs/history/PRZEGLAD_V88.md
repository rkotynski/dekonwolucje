# Przegląd zmian w wersji v88

## Znaczenie opcji zapisanej PSF dla danych eksperymentalnych

Opcja **Use stored degradation/paired PSF snapshot** ma dwa różne znaczenia
zależnie od pochodzenia danych.

### Syntetyczne zaburzanie

Po włączeniu używane jest dokładnie jądro PSF zapisane w chwili utworzenia
obrazu zaburzonego. Po wyłączeniu jądro jest ponownie przygotowywane z
bieżącej znanej PSF i aktualnej szerokości nośnika.

### Para pomiarowa „measured image + PSF”

Program nie zna rzeczywistego operatora, który wytworzył obraz pomiarowy. Przy
wspólnym wczytaniu obrazu i PSF tworzy więc dwie reprezentacje:

- `psf` — pełna, aktualnie wczytana PSF;
- `degradation_psf` — zamrożona, kompaktowa kopia przygotowana w chwili
  wczytania pary, zgodnie z ówczesnym nośnikiem PSF.

W tym przypadku słowo „degradation” jest historyczną nazwą zmiennej. Zapisana
PSF nie jest PSF używaną przez program do zaburzania obrazu, lecz ustalonym
jądrem rekonstrukcji przypisanym do pary pomiarowej.

Bezpośrednio po wczytaniu pary wynik przy włączonej i wyłączonej opcji powinien
być taki sam z dokładnością numeryczną. Zastosowanie progu PSF w zakładce 2
odświeża obecnie obie reprezentacje przez tę samą procedurę ograniczania
nośnika, więc samo progowanie również nie powinno ich rozdzielać. Różnica
pojawia się wtedy, gdy po utworzeniu zapisanej kopii zmieniono nośnik obliczeniowy,
wczytano inną PSF albo w inny sposób zmodyfikowano tylko bieżącą reprezentację.

Jeżeli obraz i PSF są wczytywane osobno, a w stanie programu nie istnieje
zapisana kopia pary, przełącznik nie ma alternatywnego jądra do wyboru i program
używa bieżącej znanej PSF niezależnie od jego zaznaczenia.

### Poprawiona niespójność po progowaniu PSF

W poprzedniej wersji po progowaniu PSF pary pomiarowej zapisane jądro mogło zostać
zastąpione pełną tablicą PSF, podczas gdy wariant bieżący ponownie ograniczał ją
do wybranego nośnika. Przełącznik mógł wtedy zmieniać wynik mimo niezmienionego
modelu fizycznego. W wersji v88 zapisana PSF po progowaniu jest ponownie
przygotowywana przez tę samą funkcję co przy wczytaniu pary.

## Kopiowanie profilu Wienera

W zakładce algorytmów, dla metody **Wiener**, dodano przycisk:

**Copy Wiener settings to all applicable algorithms**

Przycisk kopiuje zapisany profil Wienera do wszystkich algorytmów, które używają
filtru Wienera jako rekonstrukcji, inicjalizacji, kroku pośredniego lub
preconditionera. Kopiowane są:

- regularizacja `K`;
- użycie widma mocy szumu `wiener_use_noise_psd`;
- sposób pobierania wyniku odwrotnej FFT: część rzeczywista albo wartość
  bezwzględna;
- wybór zamrożonej albo bieżącej PSF dla algorytmów nieślepych.

Nie są kopiowane:

- tryb **Full loaded PSF array (circular FFT)**, ponieważ iteracyjne metody
  używają liniowego operatora z zerowymi warunkami brzegowymi;
- stan **Begin with Wiener filter**. Przycisk przenosi parametry filtru, lecz nie
  włącza automatycznie jego użycia w metodach, w których jest ono opcjonalne;
- parametry skanowania `K`, które służą tylko do prezentacji serii wyników
  zwykłego filtru Wienera.

Jeżeli Auto zoptymalizuje `K` dla zwykłego lub wsadowego filtru Wienera, wynik
jest zapisany w profilu Wienera i może zostać rozprowadzony tym przyciskiem.

## Ujednolicenie pomocniczych filtrów Wienera

Pomocnicze wywołania Wienera w Richardsonie–Lucy, Landweberze, Kaczmarzu,
metodach Rosena, Adamie i metodach ślepych honorują teraz te same ustawienia:
`K`, opcjonalne widmo mocy szumu oraz tryb `real(ifft2)`/`abs(ifft2)`.
Dotyczy to implementacji NumPy i Torch. Obliczenia Torch domyślnie pozostają w
`float32` i używają CUDA, gdy jest dostępna.

Dla danych eksperymentalnych opcja widma mocy szumu zwykle nie działa, ponieważ
wczytany obraz nie zawiera metadanych `noise_psd`. Jest użyteczna głównie dla
szumu wygenerowanego przez program.

## Weryfikacja

Uruchomiono dziesięć testów obejmujących między innymi:

- zgodność liniowego splotu i operatora sprzężonego;
- domyślny typ `float32` w Torch;
- zachowanie zamrożonej PSF pary pomiarowej;
- brak różnicy między PSF zamrożoną i bieżącą przy niezmienionych ustawieniach;
- zmianę wyniku po zmianie nośnika bieżącej PSF;
- zachowanie kompaktowego nośnika po progowaniu PSF pary pomiarowej;
- obsługę widma mocy szumu i `abs(ifft2)` przez inicjalizator Torch;
- uruchomienie wszystkich algorytmów z rejestru.

Wszystkie testy zakończyły się powodzeniem. Środowisko testowe nie zawierało
PyQt5 ani karty CUDA dostępnej dla procesu testowego, dlatego interfejsu
graficznego i rzeczywistego wykonania GPU nie uruchamiano.
