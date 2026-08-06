# Historia zmian

## 0.108.0

- Dodano przycisk **Wyczyść obrazy** w karcie 1. Usuwa obrazy, PSF, historie i wyniki bez resetowania ustawień GUI ani algorytmów.

## 0.107.0

- Dodano osobne przyciski **Wczytaj obraz zaburzony** i opcjonalny **Wczytaj obraz referencyjny**.
- Obraz eksperymentalny nie pojawia się już jako referencja po progowaniu w karcie 2.
- Obraz referencyjny jest używany wyłącznie do metryk i kryteriów Auto opartych na referencji.

## 0.106.0

- Dodano publiczny, niezależny od Qt automatyczny dobór parametrów Auto i automatyczną dekonwolucję.
- Dodano strategie przeszukiwania współrzędnych z dopasowaniem kwadratowym i ograniczonego przeszukiwania wsadowego, strojenie przez odpowiednik Torch oraz weryfikację na żądanej implementacji.
- Dodano samodzielny przykład Auto dla metody Richardsona-Lucy'ego i rozszerzono dokumentację API.

## 0.105.0

- Rozszerzono niezależne od Qt API o wszystkie 15 zarejestrowanych algorytmów i dedykowane funkcje skrótowe.
- Dodano generatory PSF Gaussa, wysokoczęstotliwościowej i niekoherentnej soczewki oraz ogólny dispatcher PSF.
- Dodano i poprawiono przykłady Richardson-Lucy, Richardson-Lucy-Wiener, Richardson-Lucy-Rosen, Landwebera i blokowej metody Kaczmarza.
- Zaktualizowano listę autorów: Amine Güneş i Rafał Kotyński, University of Warsaw, Faculty of Physics.

## 0.104.4

- Osadzono cztery końcowe zrzuty GUI bezpośrednio w głównym dwujęzycznym pliku README.
- Dodano dwujęzyczne podpisy opisujące główny przepływ pracy, przygotowanie PSF, ustawienia blokowej metody Kaczmarza i ocenę iteracji.
- Usunięto z README określenia sugerujące, że zrzuty są jedynie miejscami zastępczymi.

## 0.104.3

- Rozszerzono opis metody Richardson-Lucy-Rosen, pokazując, jak nieliniowa korelacja widmowa zastępuje klasyczną projekcję wsteczną sprzężonym operatorem Richardsona-Lucy'ego.
- Zdefiniowano różnicę między elementowym sprzężeniem widma `H*` a operatorem sprzężonym `\mathcal H*`.
- Wyjaśniono, że `L=M=1` daje korelację kołową przed normalizacją, ale nie jest dokładnie tożsame z liniowym operatorem `same` z zerowymi warunkami brzegowymi używanym w klasycznej metodzie Richardson-Lucy.

## 0.104.2

- Rozszerzono w PDF definicję liniowego splotu z zerowymi warunkami brzegowymi i jawnie powiązano zapis operatorowy z obliczeniową PSF.
- Zdefiniowano parę dwuwymiarowej DFT dokładnie zgodnie z użyciem w SciPy i PyTorch (`norm="backward"`) oraz opisano twierdzenie o splocie kołowym i zawijanie na brzegach.
- Przeniesiono informację o użyciu narzędzi LLM na koniec dokumentu PDF.

## 0.104.1

- Dodano do repozytorium i angielskiego PDF cztery dostarczone zrzuty GUI.
- Zastąpiono miejsca na ilustracje ostatecznymi podpisami odnoszącymi się do rzeczywistych widoków.
- Dodano klikalne odwołania do prac oryginalnych i klasycznych dotyczących opisanych metod numerycznych.
- Wyjaśniono, które algorytmy hybrydowe i warianty stabilizacji są specyficzne dla tej implementacji.

## 0.103.3

## 0.104.0

- Dodano publiczne API Pythona niezależne od Qt dla wszystkich zarejestrowanych algorytmów.
- Dodano funkcje konwersji, generowania obrazu testowego i ruchowej PSF, zaburzania, zapisu oraz skrót do filtru Wienera.
- Dodano kompletny samodzielny przykład Wienera z ukośną PSF ruchową.
- Dodano dwujęzyczną dokumentację API i sekcję API w PDF.

- Rozszerzono matematyczny i implementacyjny opis blokowej metody Kaczmarza.
- Dodano miejsca na zrzuty ekranu oraz proponowany zestaw ilustracji z podpisami.
- Dodano dwujęzyczne praktyczne wskazówki dotyczące metody Kaczmarza.
- Poprawiono zgodność GitHub Actions z Pythonem 3.10 i opcjonalnymi testami PyTorch.

# v103.2 - audyt tłumaczeń i informacja o użyciu AI

- Sprawdzono teksty statyczne i dynamiczne GUI oraz uzupełniono brakujące polskie tłumaczenia.
- Zastąpiono fragmentaryczne zamiany komunikatów dynamicznych tłumaczeniem pełnych szablonów, eliminując komunikaty mieszane językowo i zniekształcone wyrazy.
- Ujednolicono polskie określenie wejścia odpowiadającego angielskiemu terminowi „degraded input” na **obraz zaburzony**.
- Dodano informację o częściowym użyciu narzędzi LLM przy przygotowaniu programu i dokumentacji.
- Usunięto informację o autorze z angielskiej dokumentacji PDF.

# v103.1 - poprawka uruchamiania i dokumentacja PDF

- Dodano brakujący import `sys`, wymagany przy wyborze katalogu konfiguracji zależnie od systemu.
- Dodano angielski PDF opisujący GUI, przepływ danych, modele splotu, zaimplementowane metody, kryteria, strojenie Auto i architekturę programu.
- Dołączono PDF do paczki źródłowej GitHub oraz dokumentacji instalowanej z pliku wheel.

# Historia zmian

## v103 — dwujęzyczne GUI i projekt gotowy do GitHuba

- Dodano pełne przełączanie GUI między językiem polskim i angielskim, zapisywane w aktywnym profilu ustawień JSON.
- Dodano tłumaczone klasy widżetów obejmujące etykiety formularzy, przyciski, pola wyboru, listy, podpowiedzi, dialogi, komunikaty stanu, dzienniki i tytuły wykresów Matplotlib.
- Zachowano kanoniczne angielskie identyfikatory algorytmów, klucze konfiguracji i komentarze w kodzie.
- Dodano `pyproject.toml`, licencję MIT, metadane cytowania, testy GitHub Actions, szablony zgłoszeń i dokumentację dwujęzyczną.
- Przeniesiono domyślny plik ustawień do katalogu konfiguracji użytkownika i dodano zapamiętywanie ostatnio wybranego profilu.

## v102

- Auto zamraża opcjonalne etapy wyłączone w chwili rozpoczęcia strojenia.
- Poprawiono bezreferencyjną wspólną optymalizację progu PSF i K Wienera przez ograniczenie progu statystykami tła PSF i odrzucanie zapadniętych jąder.

## v101

- Dodano współpracujące anulowanie Auto z pięciosekundowym mechanizmem wymuszonego zakończenia izolowanego procesu.

## v100

- Progi i ramka PSF stają się danymi obliczeniowymi dopiero po naciśnięciu Apply.
- Usunięto moduł wyniku odwrotnej FFT w filtrze Wienera.

## v99 i wcześniejsze

Wcześniejsze wersje wprowadziły jedną jawną PSF obliczeniową, prostokątny wybór PSF, wsadowe kryteria iteracji, szybkie poziomy wyświetlania, spójne operatory FFT/splotu oraz rozbudowane testy regresji numerycznej. Szczegółowe historyczne notatki po polsku znajdują się w `docs/history/`.
