# Dekonwolucje 0.107.0 - niezależny obraz zaburzony i referencyjny

## Przyczyna błędu

W poprzedniej wersji przycisk `Load image` zapisywał wczytaną tablicę równocześnie jako:

- obraz wejściowy rekonstrukcji (`degraded`),
- obraz referencyjny (`image`).

Metryki rozpoznawały początkowo, że są to te same dane, lecz po progowaniu w karcie 2 obiekt wejściowy był zastępowany nową tablicą. Obraz przechowywany jako `image` pozostawał niezmieniony i był wtedy błędnie traktowany jako niezależna referencja. Dlatego po `Apply thresholds / PSF selection` pojawiał się podgląd referencji i mogły zostać włączone PSNR/SSIM.

## Nowy przepływ danych

W karcie 1 są osobne przyciski:

1. `Load disturbed image` / `Wczytaj obraz zaburzony`,
2. `Load reference image` / `Wczytaj obraz referencyjny` - opcjonalny,
3. `Load PSF`,
4. `Generate test image`,
5. `Generate selected PSF`,
6. `Generate degraded input`.

Wczytanie obrazu zaburzonego:

- tworzy wyłącznie dane wejściowe rekonstrukcji,
- usuwa poprzednią referencję,
- ustawia `reference_available=False`,
- ukrywa podgląd referencji,
- wyłącza PSNR, SSIM i referencyjne kryteria Auto.

Obraz referencyjny można wczytać później osobnym przyciskiem. Jest on używany wyłącznie do metryk i kryteriów Auto, nigdy jako wejście rekonstrukcji.

## Progowanie w karcie 2

`Apply thresholds / PSF selection` modyfikuje wyłącznie:

- obraz zaburzony używany przez algorytm,
- PSF obliczeniową po progowaniu, przycięciu i normalizacji.

Operacja nie tworzy ani nie odtwarza obrazu referencyjnego. Przy danych eksperymentalnych bez referencji karta 1, karta 2 i karta 4 nie pokazują jej po zastosowaniu progów.

## Zmiana geometrii i PSF

Przebudowano funkcję reframingu. Osobno wczytane obrazy zaburzony i referencyjny są dopełniane oraz uzgadniane niezależnie. Zmiana PSF, rozmiaru siatki albo widocznego dopełnienia nie może już skopiować obrazu zaburzonego do stanu referencji ani usunąć niezależnie wczytanego pomiaru.

## Dokumentacja i testy

Zaktualizowano README, polską i angielską instrukcję użytkownika oraz PDF. Dodano testy regresji sprawdzające:

- brak metryk referencyjnych dla samego obrazu eksperymentalnego,
- aktywację metryk po wczytaniu niezależnej referencji,
- wyłączenie metryk, gdy referencja jest identyczna z pomiarem,
- rozdzielenie ról obu przycisków w kodzie GUI.

Wynik testów: 54 testy pytest i 15 podtestów algorytmów zakończonych powodzeniem.
