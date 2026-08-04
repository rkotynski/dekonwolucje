# Przegląd zmian w wersji v102

## 1. Auto nie zmienia parametrów wyłączonych funkcji

Stan opcjonalnych etapów jest zamrażany na początku każdego przebiegu **Auto** i **Auto All**. Auto nie może już chwilowo włączyć funkcji, a następnie w kolejnej współrzędnej zmienić parametrów, które przy rozpoczęciu strojenia były ukryte lub nieaktywne.

W szczególności:

- gdy **Begin with Wiener filter** jest wyłączone, `K` i ustawienia pomocniczego Wienera nie są strojone w algorytmach, w których Wiener służy wyłącznie do inicjalizacji;
- gdy **Denoiser timing** ma wartość `Off`, nie zmieniają się rodzaj denoisera, jego siła ani ścieżka wag;
- gdy opcjonalny krok TV jest wyłączony, Auto nie zmienia jego wagi ani liczby iteracji;
- gdy relaksacja parametrów Rosena jest wyłączona, nie jest zmieniany współczynnik relaksacji;
- przełączniki aktywujące te etapy pozostają w stanie początkowym przez cały przebieg Auto.

Parametry właściwe dla algorytmu nadal są strojone. Na przykład `K` pozostaje aktywne w zwykłym filtrze Wienera, Richardsonie–Lucym–Wienerze i preconditionerze Wienera Landwebera, ponieważ w tych metodach nie jest tylko parametrem wyłączonej inicjalizacji.

## 2. Diagnoza wspólnej optymalizacji progu PSF i K

W v101 każdy kandydat PSF był poprawnie:

1. progowany;
2. przycinany do aktualnej prostokątnej ramki;
3. normalizowany do sumy 1;
4. przekształcany do OTF.

Problem dotyczył funkcji celu. GCV jest właściwym narzędziem do wyboru `K` przy ustalonym operatorze, lecz nie daje wiarygodnej podstawy do porównywania dowolnie różnych PSF. Przy jednoczesnej zmianie progu mogło preferować PSF zredukowaną do jednego lub kilku pikseli. Taka niemal impulsowa PSF często otrzymywała bardzo niską wartość GCV, mimo że nie odpowiadała zmierzonemu rozkładowi PSF.

To wyjaśnia obserwację, że wysoki próg mógł otrzymać formalnie lepsze kryterium, a optymalizacja samego `K` dla nieprogowanej PSF mogła wyglądać gorzej wizualnie mimo innej wartości kryterium. Wartości GCV dla zasadniczo różnych operatorów nie powinny być interpretowane jak bezpośrednio porównywalna miara jakości obrazu.

## 3. Nowa optymalizacja zagnieżdżona

### Z niezależnym obrazem referencyjnym

Próg PSF i `K` są nadal wybierane przez minimalizację MSE rekonstrukcji. Obecność prawdziwej referencji pozwala wiarygodnie porównywać różne kandydaty PSF.

### Bez niezależnej referencji

Procedura jest teraz zagnieżdżona:

1. z obwodu aktualnie wybranej ramki PSF obliczana jest odporna wartość tła: mediana i MAD;
2. na ich podstawie wyznaczany jest dopuszczalny przedział progu;
3. automatyczny próg bez referencji nie może przekroczyć `0.25 × peak`;
4. dla każdego dopuszczalnego progu PSF jest progowana, przycinana i normalizowana;
5. dla tak ustalonej PSF `K` jest wybierane przez GCV;
6. próg jest wybierany z uwzględnieniem zgodności z poziomem tła PSF;
7. odrzucane lub silnie karane są jądra tracące niemal całą masę albo zapadające się do kilku efektywnych pikseli.

GCV nie wybiera już swobodnie progu. Służy wyłącznie do warunkowego wyboru `K` dla jednej, ustalonej PSF.

## 4. Rozszerzona diagnostyka przycisku

Komunikat po **Optimize PSF floor + Wiener K** pokazuje teraz:

- próg przed i po optymalizacji;
- `K` przed i po optymalizacji;
- odporny poziom tła PSF i jego rozrzut;
- dopuszczalny przedział przeszukiwania progu;
- warunkową wartość GCV dla wybranej PSF;
- zachowaną część masy PSF;
- liczbę niezerowych i efektywnych pikseli;
- sumę PSF po normalizacji.

`K` jest zapisywane od razu w profilu Wienera. Próg pozostaje ustawieniem oczekującym i trafia do obliczeń po naciśnięciu **Apply thresholds / PSF selection now**.

## 5. Test kontrolny regresji

Dodano przypadek z zaszumioną PSF i niezerowym tłem, dla którego stara procedura mogła wybierać próg około `0.9 × peak` oraz jądro złożone z dwóch pikseli. Nowa procedura ogranicza próg do przedziału wynikającego z tła PSF, zachowuje rozłożone jądro i raportuje jego efektywną liczbę pikseli.

Cały zestaw zakończył się wynikiem **34 zaliczonych testów i 15 podtestów algorytmów**.
