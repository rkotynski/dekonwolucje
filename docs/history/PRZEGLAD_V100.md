# Przegląd zmian w wersji v100

## Zatwierdzanie zmian w karcie 2

Zmiana suwaka **Measured image floor**, suwaka **PSF floor / peak**, położenia ramki albo jej szerokości i wysokości nie modyfikuje już danych obliczeniowych. Są to ustawienia oczekujące.

Dopiero przycisk **Apply thresholds / PSF selection now**:

1. pobiera pierwotny obraz i PSF zapisane na początku bieżącej sesji progowania;
2. stosuje próg obrazu;
3. stosuje próg PSF;
4. zeruje pełną tablicę PSF poza wskazanym prostokątem;
5. wycina prostokątną część PSF, z dopełnieniem zerami, gdy ramka wychodzi poza tablicę;
6. normalizuje wycięte jądro do sumy równej 1;
7. aktualizuje podglądy, histogramy i dane przekazywane do algorytmów.

Histogramy przedstawiają zatem ostatnio **zatwierdzony** obraz obliczeniowy i ostatnio zatwierdzoną, przyciętą oraz znormalizowaną PSF. Podczas edycji może przesuwać się jedynie linia pokazująca oczekujący próg; rozkład histogramu pozostaje niezmieniony.

Przyciski **Auto from border** oraz **Optimize PSF floor + Wiener K** ustawiają wartości w kontrolkach, ale zmiana PSF zostaje zastosowana dopiero po naciśnięciu **Apply**. Zoptymalizowane `K` jest nadal zapisywane od razu w profilu Wienera.

## Pełny podgląd PSF po Apply

Po zastosowaniu ustawień tryb **Full PSF array** pokazuje pełną tablicę PSF po progowaniu, przy czym wszystkie próbki poza ostatnio zatwierdzoną ramką mają wartość zero. Czerwona ramka może następnie pokazywać nowy, jeszcze niezatwierdzony wybór.

Tryb **Selected calculation part** pokazuje dokładne kompaktowe jądro obliczeniowe po wycięciu i normalizacji. Jego suma wynosi 1.

## Zmiana rozmiaru ramki kółkiem myszy

W pełnym podglądzie PSF:

- obrót kółka do góry zmniejsza prostokąt;
- obrót kółka w dół zwiększa prostokąt;
- środek ramki pozostaje w tym samym miejscu;
- szerokość i wysokość są skalowane równocześnie, z przybliżonym zachowaniem proporcji prostokąta;
- zmiana pozostaje oczekująca do naciśnięcia **Apply**.

Nadal można niezależnie zmieniać szerokość i wysokość przez przeciąganie odpowiednich krawędzi.

## Usunięcie `Use |inverse FFT|`

Opcja **Use |inverse FFT|** została usunięta z karty algorytmów i ze wspólnego profilu Wienera.

Dla rzeczywistego obrazu wynik filtru Wienera jest obecnie zawsze wyznaczany jako:

\[
\widehat f=\operatorname{Re}\left\{\mathcal F^{-1}\!\left[
\frac{H^*G}{|H|^2+K N}
\right]\right\}.
\]

Zastosowanie modułu zespolonego po IFFT było nieliniową operacją dodatkową. Zamieniało ujemne oscylacje rekonstrukcji na wartości dodatnie i mogło tworzyć pozornie czystszy, lecz zmieniony ilościowo obraz.

Stary parametr `wiener_absolute_output`, jeżeli pozostał w wcześniejszym pliku konfiguracyjnym, jest usuwany podczas wczytywania profilu. Ukryty argument zgodności w funkcjach numerycznych jest ignorowany i nie zmienia wyniku.

## Weryfikacja

Zakończyło się powodzeniem 27 testów numerycznych i uruchomieniowych. Testy obejmują:

- zgodność jawnego filtru Wienera ze wzorem FFT/IFFT;
- identyczny wynik dla dawnego argumentu `absolute_output=False` i `absolute_output=True`;
- zerowanie pełnej PSF poza zatwierdzonym prostokątem;
- normalizację kompaktowej PSF po wycięciu;
- uruchomienie wszystkich zarejestrowanych algorytmów.

Kod GUI przeszedł kontrolę składni. Środowisko testowe nie zawierało PyQt5, dlatego obsługi kółka myszy nie uruchomiono w rzeczywistym oknie.
