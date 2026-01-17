# Credit Scoring Engine

Biblioteka do budowy modeli scoringowych dla oceny ryzyka kredytowego.

## Dokumentacja

### Metodologia

Szczegółowa dokumentacja metodologiczna znajduje się w pliku **[METHODOLOGICAL_README.md](METHODOLOGICAL_README.md)**, który zawiera:

- Fundament metodologiczny i literaturę referencyjną
- Metody Reject Inference (wnioskowanie o odrzuconych)
- Kalibrację i skalowanie punktacji (PDO, formuła skalowania)
- Rekomendacje implementacyjne (optbinning, scorecardpy, Toad)
- Strukturę modułów biblioteki
- Metryki i monitoring

### Materiały Referencyjne

Dodatkowe dokumenty referencyjne znajdują się w folderze `docs/references/`:

| Dokument | Opis |
|----------|------|
| [literature.md](docs/references/literature.md) | Pełna lista literatury i papers akademickich |
| [reject_inference.md](docs/references/reject_inference.md) | Szczegółowa dokumentacja metod Reject Inference |
| [calibration.md](docs/references/calibration.md) | Dokumentacja kalibracji i skalowania punktacji |
| [metrics.md](docs/references/metrics.md) | Dokumentacja metryk (Gini, KS, PSI, IV, etc.) |

## Kluczowe Funkcjonalności

### Moduły (Planowane)

1. **Binning & WoE** - Dyskretyzacja zmiennych i Weight of Evidence
2. **Feature Selection** - Selekcja zmiennych (IV, VIF, Stepwise)
3. **Modeling** - Wrapper na regresję logistyczną z obsługą wag
4. **Calibration** - Skalowanie prawdopodobieństwa na punktację
5. **Monitoring** - PSI, CSI i metryki performance

## Szybki Start

```python
# Przykład użycia (planowany)
from credit_scoring import Binning, FeatureSelector, Scorecard

# Binning
binner = Binning(method='optimal', monotonic=True)
binned_data = binner.fit_transform(X, y)

# Selekcja zmiennych
selector = FeatureSelector(iv_min=0.02, vif_max=5.0)
selected_features = selector.fit_transform(X, y)

# Budowa scorecardu
scorecard = Scorecard(base_score=600, base_odds=50, pdo=20)
scorecard.fit(X, y)
scores = scorecard.predict(X_new)
```

## Literatura Referencyjjna

Główne pozycje literaturowe dla credit scoringu:

1. **Siddiqi, N. (2017)** - "Intelligent Credit Scoring" - Absolutna podstawa
2. **Thomas, L. C. et al. (2002)** - "Credit Scoring and Its Applications" - Akademickie podejście
3. **Hand, D. J. & Henley, W. E. (1997)** - Seminal paper porównujący metody

Pełna lista w [docs/references/literature.md](docs/references/literature.md)

## Licencja

MIT License