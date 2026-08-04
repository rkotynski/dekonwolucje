# Dekonwolucje / Deconvolution GUI

[English](#english) · [Polski](#polski)

## English

A desktop research application for grayscale-image deconvolution with a known or unknown point-spread function (PSF). The GUI provides Wiener, Richardson–Lucy, Rosen, Landweber, Kaczmarz, Adam TV-MAP and blind-deconvolution variants, with optional PyTorch/CUDA acceleration.

The interface can be switched at runtime between **English** and **Polish** using **Language / Język** in the menu. The selected language is stored in the active JSON settings profile. Source-code comments, configuration keys and algorithm identifiers remain in English.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
dekonwolucje
```

Optional PyTorch support:

```bash
python -m pip install -e ".[gpu]"
```

For a CUDA-enabled PyTorch build, install the wheel appropriate for the local CUDA/driver environment and then install this project.

Alternative launch commands:

```bash
python -m deconv
python run_deconvolution_gui.py
```

### Documentation

- [English PDF: program description and numerical methods](docs/Deconvolution_GUI_and_Methods_EN.pdf)
- [English user guide](docs/USER_GUIDE_EN.md)
- [English architecture and developer guide](docs/ARCHITECTURE_EN.md)
- [English changelog](docs/CHANGELOG_EN.md)
- [Polish user guide](docs/USER_GUIDE_PL.md)
- [Polish architecture and developer guide](docs/ARCHITECTURE_PL.md)
- [Polish changelog](docs/CHANGELOG_PL.md)

### AI-assisted development disclosure

Parts of the software and documentation were prepared with assistance from tools based on large language models (LLMs). Their suggestions were incorporated as part of the development process; numerical methods, implementation details and results should nevertheless be independently verified for the intended scientific application.

### Testing

```bash
python -m pip install -e ".[test]"
pytest
```

### License and author

MIT License. Copyright © 2026 **Rafał Kotyński**, University of Warsaw, Faculty of Physics.

Homepage: <https://github.com/rkotynski/dekonwolucje>

---

## Polski

Aplikacja badawcza z interfejsem graficznym do dekonwolucji obrazów w skali szarości ze znaną lub nieznaną punktową funkcją rozmycia (PSF). Program zawiera warianty metod Wienera, Richardsona–Lucy'ego, Rosena, Landwebera, Kaczmarza, Adam TV-MAP i dekonwolucji ślepej, z opcjonalnym przyspieszeniem PyTorch/CUDA.

Język interfejsu można przełączać podczas działania programu między **polskim** i **angielskim** w menu **Język / Language**. Wybór jest zapisywany w aktywnym profilu ustawień JSON. Komentarze w kodzie, klucze konfiguracji i identyfikatory algorytmów pozostają angielskie.

### Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
dekonwolucje
```

Opcjonalna obsługa PyTorch:

```bash
python -m pip install -e ".[gpu]"
```

Dla PyTorch z CUDA należy najpierw zainstalować pakiet odpowiedni dla lokalnego środowiska CUDA i sterownika, a następnie zainstalować projekt.

Alternatywne sposoby uruchomienia:

```bash
python -m deconv
python run_deconvolution_gui.py
```

### Dokumentacja

- [Angielski PDF: opis programu i metod numerycznych](docs/Deconvolution_GUI_and_Methods_EN.pdf)
- [Instrukcja użytkownika po polsku](docs/USER_GUIDE_PL.md)
- [Architektura i przewodnik programisty po polsku](docs/ARCHITECTURE_PL.md)
- [Historia zmian po polsku](docs/CHANGELOG_PL.md)
- [English user guide](docs/USER_GUIDE_EN.md)
- [English architecture and developer guide](docs/ARCHITECTURE_EN.md)
- [English changelog](docs/CHANGELOG_EN.md)

### Informacja o użyciu narzędzi AI

Przy przygotowaniu części programu i dokumentacji korzystano z narzędzi opartych na dużych modelach językowych (LLM). Ich sugestie włączano w ramach procesu tworzenia projektu; metody numeryczne, szczegóły implementacji i wyniki należy jednak niezależnie zweryfikować dla zamierzonego zastosowania naukowego.

### Testy

```bash
python -m pip install -e ".[test]"
pytest
```

### Licencja i autor

Licencja MIT. Copyright © 2026 **Rafał Kotyński**, Wydział Fizyki Uniwersytetu Warszawskiego.

Strona projektu: <https://github.com/rkotynski/dekonwolucje>
