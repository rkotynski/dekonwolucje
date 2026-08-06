# Instrukcja użytkownika

## 1. Uruchamianie programu

Uruchom `dekonwolucje`, `python -m deconv` albo `python run_deconvolution_gui.py`. Domyślne ustawienia są przechowywane w katalogu konfiguracji użytkownika (`~/.config/dekonwolucje` w systemie Linux, jeśli nie ustawiono `XDG_CONFIG_HOME`). Inny profil JSON można wybrać w menu **Ustawienia**; ostatnio wybrany profil jest zapamiętywany.

Menu **Język → Polski/Angielski** przełącza język całego GUI. Zmiana jest natychmiastowa i nie modyfikuje obrazów, parametrów ani profili algorytmów.

## 2. Karta 1 — Obraz i PSF

Pierwsza karta służy do wczytywania lub generowania obrazu źródłowego i PSF. Podglądy oraz histogramy pokazują bieżące dane obliczeniowe, a nie nieaktualną kopię źródłową.

Kolejność przycisków:

1. **Wczytaj obraz zaburzony**
2. **Wczytaj obraz referencyjny** (opcjonalnie)
3. **Wczytaj PSF**
4. **Wygeneruj obraz testowy**
5. **Wygeneruj wybraną PSF**
6. **Wygeneruj obraz zaburzony**
7. **Wyczyść obrazy**

Dla danych eksperymentalnych przycisk **Wczytaj obraz zaburzony** tworzy wyłącznie dane wejściowe rekonstrukcji. Nie tworzy ani nie duplikuje obrazu referencyjnego. Niezależny obraz prawdziwy można wczytać osobno przyciskiem **Wczytaj obraz referencyjny**; jest on używany tylko do PSNR/SSIM i kryteriów Auto opartych na referencji, nigdy jako wejście rekonstrukcji. Gdy referencji nie wczytano, jej podgląd pozostaje ukryty, a metryki referencyjne są wyłączone.

**Wyczyść obrazy** usuwa wczytane lub wygenerowane obrazy, PSF, historie rekonstrukcji i wyniki, ale zachowuje bieżące ustawienia obliczeń, GUI i algorytmów.

Jeżeli tablice obrazu i PSF mają różne wymiary, mniejsza tablica jest centrycznie dopełniana zerami do wspólnego płótna. Piksele nie są skalowane ani przycinane.

## 3. Karta 2 — Progi i PSF obliczeniowa

Karta 2 definiuje dane rzeczywiście przekazywane do każdego algorytmu.

- Próg obrazu zeruje niskie wartości i przeskalowuje pozostały zakres.
- Próg PSF jest podawany względem maksimum PSF.
- Czerwona ramka wybiera prostokątny fragment PSF.
- Ramkę można przesuwać, zmieniać przez przeciąganie krawędzi lub narożnika oraz skalować kółkiem myszy wokół jej środka.
- **Zastosuj progi / wybór PSF** zatwierdza oczekujące ustawienia.
- **Resetuj progi / wybór PSF** przywraca tablice źródłowe wczytane lub wygenerowane.

Po zatwierdzeniu wartości poza zaakceptowaną ramką są zerowe w podglądzie pełnej tablicy PSF. Wybrane kompaktowe jądro jest przycinane, rzutowane na wartości nieujemne i normalizowane tak, aby jego suma była równa jeden. Histogramy zmieniają się dopiero po zatwierdzeniu.

Przycisk **Optymalizuj próg PSF + K Wienera** proponuje jednocześnie próg PSF i regularyzację Wienera. Przy obrazie referencyjnym minimalizowany jest MSE rekonstrukcji. Bez referencji dopuszczalny próg jest ograniczony odpornymi statystykami tła PSF, a GCV wybiera K dla każdej ustalonej kandydatury PSF.

## 4. Karta 3 — Algorytm

Wybierz algorytm i jego parametry. Opcjonalne etapy przetwarzania są jawne. Auto zamraża stan ich aktywacji na początku strojenia: parametry wyłączonej inicjalizacji Wienera, odszumiacza, kroku TV albo relaksacji Rosena nie są zmieniane.

Opcja **Wsadowy PyTorch** wybiera implementację wsadową, gdy jest dostępna. Obliczenia Torch domyślnie używają `float32`; CUDA jest używana tylko po jej wybraniu i gdy jest dostępna.

Metody ślepe pobierają szerokość i wysokość estymowanej PSF bezpośrednio z ramki w karcie 2.

### Metoda blokowa Kaczmarza w praktyce

**Blokowa metoda Kaczmarza** jest eksperymentalną metodą dekonwolucji typu ART. Dzieli płaszczyznę obrazu pomiarowego na kwadratowe bloki, wyznacza reszty tylko w wybranych blokach, łączy je z opcjonalnym nakładaniem i gładkim ważeniem, a następnie propaguje połączoną resztę wstecz przez operator sprzężony PSF. Program nie buduje jawnie macierzy splotu, a aktualizacja nie jest dokładnym rzutem blokowym.

Zalecany punkt początkowy:

- pozostaw włączone **Pełny przebieg**, **Nakładające się bloki**, **Przesuwana siatka**, **Gładkie okno bloków** i **Stabilizowany przebieg**;
- zacznij od rozmiaru bloku 32, relaksacji 0,15, tłumienia 0,5 i maksymalnego udziału aktualizacji 0,25;
- zmniejsz relaksację lub tłumienie, gdy kolejne klatki stają się naprzemiennie zbyt ciemne i zbyt jasne;
- zwiększ rozmiar bloku, gdy dominują szwy lub lokalna niespójność, a zmniejsz go, gdy potrzebna jest bardziej lokalna korekcja;
- porównuj automatycznie wybraną najlepszą iterację, zamiast zakładać, że najlepsza jest iteracja ostatnia.

Liczba bloków jest używana tylko po wyłączeniu **Pełnego przebiegu**. Losowa kolejność może ograniczać systematyczny wpływ kolejności, a przesuwanie siatki zmniejsza stałe granice pionowe i poziome. Opcjonalne TV oraz odszumianie są wykonywane po każdej zewnętrznej aktualizacji Kaczmarza.

## 5. Karta 4 — Test i historia wyników

Uruchom dekonwolucję i przeglądaj zapisane iteracje. Kryteria wszystkich klatek są obliczane wsadowo; w miarę możliwości wykorzystywane jest Torch/CUDA. Po zakończeniu automatycznie wybierana jest najlepsza klatka i wykonywane są **Poziomy Auto**.

Suwaki czerni i bieli obejmują pełny znormalizowany zakres i zachowują niezerowy odstęp. Zmieniają wyłącznie sposób wyświetlania, a nie zapisane wyniki numeryczne. **Wybierz najlepszą iterację** ponawia automatyczny wybór bez uruchamiania algorytmu.

## 6. Implementacja filtru Wienera

Każdy etap Wienera używa jawnych operacji FFT/IFFT. Wynikiem jest część rzeczywista odwrotnej FFT. Usunięto dawny moduł wyniku IFFT i możliwość wyboru zapisanej kopii PSF użytej do zaburzania.

## 7. Anulowanie Auto

**Anuluj Auto** najpierw zgłasza współpracujące zatrzymanie. Jeżeli bieżąca iteracja numeryczna nie zwróci sterowania w ciągu pięciu sekund, izolowany proces Auto jest kończony, natomiast proces GUI pozostaje aktywny.

## 8. Używanie algorytmów bez GUI

Publiczne API niezależne od Qt opisano w `docs/API_PL.md`. Kompletny przykład `examples/wiener_motion_blur.py` generuje standardowy obraz testowy, ukośną PSF ruchową, obraz zaburzony i rekonstrukcję filtrem Wienera.

## Informacja o użyciu narzędzi AI

Przy przygotowaniu części programu i dokumentacji korzystano z narzędzi opartych na dużych modelach językowych (LLM). Ich sugestie włączano w ramach procesu tworzenia projektu; metody numeryczne, szczegóły implementacji i wyniki należy jednak niezależnie zweryfikować dla zamierzonego zastosowania naukowego.


## Proponowane zrzuty ekranu

Umieść poniższe pliki w katalogu `docs/screenshots/`; źródło LaTeX dokumentu PDF wstawi je automatycznie, a przy ich braku pokaże przygotowane ramki.

1. `01-gui-overview.png` — **Główny przepływ pracy programu.** Okno główne z reprezentatywnym obrazem i PSF, widocznymi czterema kartami oraz opcjonalnie otwartym menu Język.
2. `02-psf-preparation.png` — **Przygotowanie PSF obliczeniowej.** Karta 2 w widoku pełnej tablicy, z dwoma histogramami i czerwoną prostokątną ramką.
3. `03-kaczmarz-settings.png` — **Ustawienia blokowej metody Kaczmarza.** Karta 3 z geometrią bloków, kolejnością, stabilizacją, tłumieniem i ograniczeniem aktualizacji.
4. `04-result-history.png` — **Ocena iteracji.** Karta 4 po obliczeniach wieloiteracyjnych, z kryteriami, wybraną najlepszą iteracją i poziomami wyświetlania.

Najlepiej użyć plików PNG, usunąć z kadru elementy pulpitu, nie pokazywać poufnych ścieżek plików i zastosować ten sam reprezentatywny zestaw danych we wszystkich ilustracjach. Szerokość około 1600–2200 pikseli jest wystarczająca do dokumentu PDF.
