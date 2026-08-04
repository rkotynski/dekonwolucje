# Przegląd zmian w wersji v101

## Cel zmiany

Przycisk **Cancel Auto** zatrzymywał dotychczas wyszukiwanie dopiero po zakończeniu bieżącego kandydata albo całej partii kandydatów. Jeżeli pojedyncze wywołanie algorytmu, iteracja Torch/CUDA lub duża partia FFT trwały długo, interfejs mógł przez wiele sekund lub minut pozostawać w stanie „cancellation requested”.

Wersja v101 wprowadza limit 5 sekund liczony od naciśnięcia **Cancel Auto**.

## Dwuetapowe anulowanie

Po naciśnięciu przycisku program:

1. natychmiast ustawia współdzielone żądanie zatrzymania;
2. algorytmy iteracyjne kończą bieżącą iterację i nie rozpoczynają następnej;
3. jeżeli obliczenie nie zwróci sterowania w ciągu 5 sekund, proces numeryczny Auto jest wymuszenie kończony.

Komunikat w karcie 3 informuje, czy anulowanie zakończyło się współpracująco, czy konieczne było wymuszone zatrzymanie po 5 sekundach.

## Dlaczego nie zastosowano `QThread.terminate()`

Asynchroniczne zakończenie wątku wykonującego kod Pythona lub CUDA jest niebezpieczne. Wątek może zostać przerwany podczas trzymania blokady, alokacji pamięci GPU albo operacji bibliotecznej. Mogłoby to pozostawić program w stanie, w którym kolejne obliczenia nie uruchamiają się albo wymagają zamknięcia aplikacji.

Dlatego obliczenia Auto zostały przeniesione do osobnego procesu:

- proces Qt i GUI pozostają w procesie głównym;
- `AutoTuneWorker` generuje kandydatów i zarządza wyszukiwaniem;
- jeden trwały proces pomocniczy wykonuje wszystkie oceny numeryczne;
- po przekroczeniu limitu kończony jest wyłącznie proces pomocniczy;
- blokada obliczeń procesu głównego jest zwalniana normalnie przez wątek sterujący.

Proces jest uruchamiany tylko raz dla całego **Auto** lub **Auto All**, więc nie ma kosztu inicjalizacji Pythona i CUDA dla każdego kandydata.

## Zatrzymywanie między iteracjami

Współdzielone żądanie zatrzymania jest przekazywane również do wsadowych implementacji Torch:

- Torch batch Richardson–Lucy;
- Torch batch Richardson–Lucy–Wiener;
- Torch batch Richardson–Lucy–Rosen;
- Torch batch Landweber;
- PyTorch Adam TV-MAP;
- PyTorch Blind Adam TV-MAP.

Implementacje sprawdzają żądanie przed rozpoczęciem kolejnej iteracji. Jeżeli pojedyncza iteracja sama trwa dłużej niż 5 sekund, proces pomocniczy jest kończony przez mechanizm limitu czasu.

Dla jednorazowych operacji, takich jak duża FFT Wienera, nie istnieje bezpieczny punkt przerwania wewnątrz wywołania bibliotecznego. W takim przypadku po 5 sekundach działa wymuszone zakończenie procesu pomocniczego.

## Zachowanie wyników

Po anulowaniu:

- niedokończony kandydat nie jest przyjmowany;
- parametry niedokończonego algorytmu nie są kopiowane do profilu;
- algorytmy ukończone wcześniej przez **Auto All** zachowują już zaakceptowane ustawienia;
- przyciski **Auto** i **Auto All** są ponownie odblokowywane po zamknięciu wątku sterującego;
- można uruchomić kolejne obliczenia bez pozostawionej blokady numerycznej.

## Testy

Dodano testy obejmujące:

- współpracujące zatrzymanie procesu przed upływem limitu;
- wymuszone zatrzymanie nieodpowiadającego zadania po upływie okresu ochronnego;
- wykonanie rzeczywistej oceny kandydata Wienera w procesie pomocniczym;
- dotychczasowe testy numeryczne i testy uruchomieniowe wszystkich algorytmów.

Łączny wynik: **32 testy zakończone powodzeniem oraz 15 podtestów algorytmów**.
