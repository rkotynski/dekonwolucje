# Przegląd zmian — wersja v99

## Jednoznaczne dane obliczeniowe

Program przechowuje obecnie jeden jawny obraz wejściowy do rekonstrukcji (`state["degraded"]`) oraz jedną PSF obliczeniową (`state["calculation_psf"]`). PSF obliczeniowa powstaje zawsze w tej kolejności:

1. zastosowanie progu do pełnej PSF;
2. wycięcie prostokąta wskazanego w karcie 2;
3. dopełnienie zerami, gdy ramka wychodzi poza tablicę;
4. usunięcie wartości ujemnych;
5. normalizacja wyciętego jądra do sumy 1.

Ten sam obiekt jest używany przez rekonstrukcję, syntetyczne zaburzanie, automatyczne strojenie, kryteria ponownego rozmycia oraz podgląd i zapis PSF użytej w obliczeniach.

## Karty 1 i 2

Obie karty pokazują:

- obraz, który zostanie przekazany do algorytmu po progowaniu;
- PSF po progowaniu, przycięciu i normalizacji;
- histogram obrazu i histogram PSF z 256 przedziałami;
- rzeczywistą rozdzielczość obrazu i PSF oraz sumę jądra PSF.

Pełna PSF pozostaje dostępna w karcie 2 wyłącznie jako widok edycyjny z czerwoną ramką. Tytuł podglądu wyraźnie informuje, że do obliczeń trafia tylko zaznaczony prostokąt. Zmiany suwaków, środka i ramki są stosowane po krótkim opóźnieniu i odświeżają również kartę 1.

Przycisk **Reset thresholds / PSF selection** jest jedyną operacją przywracającą obraz i PSF wczytane z dysku albo pierwotnie wygenerowane.

## Usunięcie zapisanej PSF użytej do zaburzania z algorytmów

Usunięto z interfejsu, profili algorytmów i aktywnych parametrów opcję:

**Use stored degradation/paired PSF snapshot**

Usunięto także wybór pełnoekranowej PSF dla zwykłego Wienera. Dawne pola są usuwane przy wczytywaniu i ponownym zapisie starszego profilu JSON. Zapisana PSF użytej do zaburzania może pozostać w stanie wyłącznie jako informacja diagnostyczna o wcześniejszej syntetycznego zaburzania, ale nie jest już możliwym wejściem rekonstrukcji.

Zachowano zgodność funkcji `reconstruction_psf_for_image()` dla zewnętrznych wywołań. Jej dawne argumenty wyboru kopii PSF są ignorowane, a funkcja zawsze deleguje do bieżącej PSF obliczeniowej.

## Filtr Wienera

Wszystkie warianty Wienera używają jawnych operacji FFT i IFFT:

- NumPy/SciPy FFT: `fft2`, utworzenie transmitancji Wienera i `ifft2`;
- Torch/CUDA: `torch.fft.fft2` i `torch.fft.ifft2`.

Nie jest wywoływana żadna dedykowana funkcja biblioteczna do dekonwolucji Wienera. Dotyczy to algorytmu Wiener, wariantu wsadowego Torch oraz filtrów Wienera używanych do inicjalizacji albo między iteracjami innych metod. NumPy używa domyślnie `float32`.

## Rozmiar PSF w metodach ślepych

Usunięto z karty 3 osobne pola szerokości roboczej i maksymalnej szerokości ślepej PSF. Obie metody ślepe otrzymują przy każdym uruchomieniu:

- `blind_psf_width` równą szerokości czerwonej ramki z karty 2;
- `blind_psf_height` równą wysokości tej ramki.

Dotyczy to także automatycznego strojenia. PSF może być inicjalizowana bieżącą PSF obliczeniową albo funkcją Gaussa, ale rozmiar estymowanej tablicy pozostaje dokładnie zgodny z kartą 2.

## Testy

Zakończone powodzeniem:

- 27 testów numerycznych;
- 15 podtestów wszystkich zarejestrowanych algorytmów;
- kontrola składni i kompilacja wszystkich modułów.

Nowe testy sprawdzają między innymi, że dawny wybór zapisanej PSF nie zmienia wyniku, prostokątny wycinek jest normalizowany po progowaniu, metody ślepe zachowują rozmiar z karty 2, a wynik Wienera jest zgodny z bezpośrednio zapisanym wzorem FFT/IFFT.

Środowisko testowe nie zawierało PyQt5, dlatego nie uruchomiono interaktywnego okna GUI. Logika numeryczna, połączenia źródłowe interfejsu i składnia modułów zostały sprawdzone automatycznie.
