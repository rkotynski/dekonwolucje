# Przegląd zmian w wersji v95

## 1. Zapamiętywanie katalogu wczytywanych danych

W aktywnym pliku konfiguracyjnym JSON zapisywane jest nowe pole:

```json
"last_image_directory": "/ścieżka/do/katalogu"
```

Pole należy do sekcji `load_generate`. Jest aktualizowane po wybraniu pliku w
oknach:

- **Load image**,
- **Load PSF image**,
- **Load measured image + PSF** — osobno po wyborze obrazu i pasującej PSF.

Po następnym uruchomieniu programu wszystkie te okna rozpoczynają przeglądanie
od zapisanego katalogu. Jeżeli katalog przestał istnieć, program używa katalogu
domowego użytkownika. Ustawienie jest przechowywane w aktualnie wybranym profilu
konfiguracyjnym, a więc różne profile mogą mieć różne katalogi robocze.

## 2. Przyczyna błędnego podglądu „Full PSF array”

Dane PSF nie były w tym miejscu faktycznie przycinane. Problem wynikał z
mechanizmu szybkiego odświeżania Matplotlib. Po pierwszym utworzeniu obrazu
`imshow()` zapamiętuje jego zasięg przestrzenny. Późniejsze wywołanie
`AxesImage.set_data()` zmienia tablicę, lecz nie zmienia tego zasięgu.

Jeżeli najpierw pokazano mały wycinek, a następnie przełączono się na pełną PSF,
pełna tablica była nadal rysowana w prostokącie o rozmiarze poprzedniego wycinka.
Osie i czerwona ramka miały już rozmiar pełnej PSF. Skutkiem był obraz w lewym
górnym obszarze, biały fragment płótna dookoła oraz pozorna niezgodność ramki z
PSF.

## 3. Poprawione odwzorowanie pikseli

Po każdej zmianie danych `ImageCanvas.show_image()` ustawia teraz jawnie zasięg:

```python
extent = (-0.5, width - 0.5, height - 0.5, -0.5)
image_artist.set_extent(extent)
```

Ten sam zakres jest stosowany do granic osi. Obraz i czerwona przerywana ramka
korzystają więc dokładnie z tych samych współrzędnych pikselowych. Dodano także
`interpolation="nearest"`, aby interpolacja wyświetlania nie przesuwała optycznie
granic małego okna.

## 4. Znaczenie trybów podglądu

- **Full PSF array** pokazuje pełną bieżącą tablicę PSF. Na tym etapie nie jest
  ona przycinana do czerwonej ramki.
- **Selected calculation part** pokazuje wybrane kwadratowe okno.
- Przygotowanie jądra do splotu wycina okno dopiero przed obliczeniami. Gdy okno
  wychodzi poza tablicę źródłową, brakujące próbki są dopełniane zerami. Ponieważ
  zero w skali szarości jest czarne, dopełnienie nie powinno wyglądać jak białe
  tło.

## 5. Weryfikacja

- wszystkie 17 testów numerycznych i uruchomieniowych zakończyło się powodzeniem;
- wszystkie moduły przeszły kontrolę składni;
- środowisko testowe nie zawierało PyQt5, dlatego interfejs nie został uruchomiony
  interaktywnie. Przyczyna błędu została jednak jednoznacznie odtworzona na
  poziomie zachowania `AxesImage.set_data()` i usunięta przez jawne ustawianie
  zasięgu obrazu.
