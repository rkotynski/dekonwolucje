# Instrukcja użytkownika

## 1. Uruchamianie programu

Uruchom `dekonwolucje`, `python -m deconv` albo `python run_deconvolution_gui.py`. Domyślne ustawienia są przechowywane w katalogu konfiguracji użytkownika (`~/.config/dekonwolucje` w systemie Linux, jeśli nie ustawiono `XDG_CONFIG_HOME`). Inny profil JSON można wybrać w menu **Ustawienia**; ostatnio wybrany profil jest zapamiętywany.

Menu **Język → Polski/Angielski** przełącza język całego GUI. Zmiana jest natychmiastowa i nie modyfikuje obrazów, parametrów ani profili algorytmów.

## 2. Karta 1 — Obraz i PSF

Pierwsza karta służy do wczytywania lub generowania obrazu źródłowego i PSF. Podglądy oraz histogramy pokazują bieżące dane obliczeniowe, a nie nieaktualną kopię źródłową.

Kolejność przycisków:

1. **Wczytaj obraz**
2. **Wczytaj PSF**
3. **Wygeneruj obraz testowy**
4. **Wygeneruj wybraną PSF**
5. **Wygeneruj obraz zaburzony**

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

## 5. Karta 4 — Test i historia wyników

Uruchom dekonwolucję i przeglądaj zapisane iteracje. Kryteria wszystkich klatek są obliczane wsadowo; w miarę możliwości wykorzystywane jest Torch/CUDA. Po zakończeniu automatycznie wybierana jest najlepsza klatka i wykonywane są **Poziomy Auto**.

Suwaki czerni i bieli obejmują pełny znormalizowany zakres i zachowują niezerowy odstęp. Zmieniają wyłącznie sposób wyświetlania, a nie zapisane wyniki numeryczne. **Wybierz najlepszą iterację** ponawia automatyczny wybór bez uruchamiania algorytmu.

## 6. Implementacja filtru Wienera

Każdy etap Wienera używa jawnych operacji FFT/IFFT. Wynikiem jest część rzeczywista odwrotnej FFT. Usunięto dawny moduł wyniku IFFT i możliwość wyboru zapisanej kopii PSF użytej do zaburzania.

## 7. Anulowanie Auto

**Anuluj Auto** najpierw zgłasza współpracujące zatrzymanie. Jeżeli bieżąca iteracja numeryczna nie zwróci sterowania w ciągu pięciu sekund, izolowany proces Auto jest kończony, natomiast proces GUI pozostaje aktywny.

## Informacja o użyciu narzędzi AI

Przy przygotowaniu części programu i dokumentacji korzystano z narzędzi opartych na dużych modelach językowych (LLM). Ich sugestie włączano w ramach procesu tworzenia projektu; metody numeryczne, szczegóły implementacji i wyniki należy jednak niezależnie zweryfikować dla zamierzonego zastosowania naukowego.

