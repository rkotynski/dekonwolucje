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
