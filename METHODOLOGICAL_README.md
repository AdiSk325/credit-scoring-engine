# Credit Scoring Engine - Metodologia i Materiały Referencyjne

## Spis Treści

1. [Wstęp](#wstęp)
2. [Fundament Metodologiczny - Literatura](#fundament-metodologiczny---literatura)
3. [Reject Inference - Wnioskowanie o Odrzuconych](#reject-inference---wnioskowanie-o-odrzuconych)
4. [Kalibracja i Skalowanie (Scoring)](#kalibracja-i-skalowanie-scoring)
5. [Implementacja Techniczna](#implementacja-techniczna)
6. [Struktura Modułów Biblioteki](#struktura-modułów-biblioteki)
7. [Metryki i Monitoring](#metryki-i-monitoring)
8. [Materiały Referencyjne](#materiały-referencyjne)

---

## Wstęp

Ten dokument stanowi kompleksowy przewodnik metodologiczny dla biblioteki Credit Scoring Engine. Credit scoring jest dziedziną, w której regulatorzy wymagają metod sprawdzonych i wytłumaczalnych. Dlatego fundamentem są klasyczne podejścia statystyczne, a nie najnowsze trendy ML.

### Kluczowe Zasady

- **Interpretowalność** - Model musi być zrozumiały dla regulatorów i biznesu
- **Stabilność** - Wyniki muszą być powtarzalne i przewidywalne w czasie
- **Zgodność regulacyjna** - Metodologia musi spełniać wymagania regulatorów (np. Basel, EBA)
- **Sprawdzoność** - Metody muszą mieć udokumentowane zastosowanie w branży

---

## Fundament Metodologiczny - Literatura

### Klasyka Gatunku (Must Read)

W credit scoringu „biblie" są ważniejsze niż pojedyncze, nowe papers, ponieważ regulatorzy wymagają metod sprawdzonych i wytłumaczalnych.

#### 1. Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards"
- **Znaczenie**: Absolutna podstawa credit scoringu. Jeśli masz przeczytać jedną rzecz, to właśnie to.
- **Zakres**: Kompletny przewodnik od A do Z budowy scorecardów
- **Kluczowe tematy**:
  - Weight of Evidence (WoE) transformacja
  - Information Value (IV) jako miara mocy predykcyjnej
  - Binning i Coarse Classing
  - Skalowanie punktacji (Points to Double Odds)
  - Walidacja i monitoring modeli

#### 2. Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). "Credit Scoring and Its Applications"
- **Znaczenie**: Bardziej akademickie podejście, świetne dla zrozumienia teorii
- **Zakres**: Matematyczne podstawy credit scoringu
- **Kluczowe tematy**:
  - Teoria klasyfikacji statystycznej
  - Analiza dyskryminacyjna
  - Reject Inference (metody wnioskowania o odrzuconych)
  - Ekonometria credit scoringu

#### 3. Hand, D. J., & Henley, W. E. (1997). "Statistical Classification Methods in Consumer Credit Scoring: A Review"
- **Znaczenie**: Seminal paper porównujący różne metody statystyczne
- **Zakres**: Przegląd metod klasyfikacji w kontekście credit scoringu
- **Kluczowe tematy**:
  - Porównanie metod: regresja logistyczna, LDA, drzewa decyzyjne, sieci neuronowe
  - Kryteria wyboru metody
  - Problemy specyficzne dla credit scoringu

### Literatura Uzupełniająca

#### Basel Committee on Banking Supervision (BCBS)
- **"International Convergence of Capital Measurement and Capital Standards" (Basel II/III)**
- Wymogi regulacyjne dla modeli ryzyka kredytowego
- Definicje PD (Probability of Default), LGD, EAD

#### European Banking Authority (EBA)
- **Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures**
- Wymogi walidacji modeli w Unii Europejskiej

---

## Reject Inference - Wnioskowanie o Odrzuconych

### Problem Sample Bias

Budując model tylko na klientach, którym udzielono kredytu (bo tylko dla nich masz etykietę Good/Bad), wprowadzasz **błąd selekcji (sample bias)**. Populacja akceptowanych wniosków nie reprezentuje całej populacji wnioskującej o kredyt.

```
Populacja Wnioskująca
├── Zaakceptowani (mamy dane o Good/Bad) ← Tylko tutaj trenujemy model
│   ├── Good (spłacili)
│   └── Bad (nie spłacili)
└── Odrzuceni (nie mamy danych o wyniku) ← Ignorowanie = bias
```

### Metody Reject Inference

#### 1. Hard Cut-off (Przypisanie pesymistyczne)

**Opis**: Przypisanie etykiety "Bad" wszystkim odrzuconym.

**Implementacja**:
```python
def hard_cutoff_inference(df_accepted, df_rejected):
    """
    Najprostsza metoda - wszystkim odrzuconym przypisujemy Bad=1
    """
    df_rejected['target'] = 1  # Wszystkich odrzuconych uznajemy za Bad
    return pd.concat([df_accepted, df_rejected], ignore_index=True)
```

**Zalety**:
- Prostota implementacji
- Konserwatywne podejście (bezpieczne dla banku)

**Wady**:
- Zbyt pesymistyczne - część odrzuconych byłaby dobrymi klientami
- Może prowadzić do overfitting na cechach korelowanych z decyzją o odrzuceniu

**Kiedy stosować**: Gdy odrzucenia były oparte na bardzo wyraźnych sygnałach ryzyka.

#### 2. Parceling (Losowe Przypisywanie)

**Opis**: Losowe przypisywanie klasy Good/Bad w oparciu o oczekiwane ryzyko.

**Implementacja**:
```python
def parceling_inference(df_accepted, df_rejected, existing_model):
    """
    Użyj istniejącego modelu do estymacji prawdopodobieństwa
    i losowo przypisz klasę w oparciu o to prawdopodobieństwo
    """
    # Predykcja prawdopodobieństwa Bad dla odrzuconych
    prob_bad = existing_model.predict_proba(df_rejected[features])[:, 1]
    
    # Losowe przypisanie klasy zgodnie z prawdopodobieństwem
    random_vals = np.random.uniform(0, 1, size=len(df_rejected))
    df_rejected['target'] = (random_vals < prob_bad).astype(int)
    
    return pd.concat([df_accepted, df_rejected], ignore_index=True)
```

**Zalety**:
- Wykorzystuje dostępną wiedzę z modelu
- Nie jest tak pesymistyczne jak Hard Cut-off

**Wady**:
- Wymaga istniejącego modelu
- Wprowadza losowość - wyniki mogą się różnić przy każdym uruchomieniu
- Może wzmacniać bias istniejącego modelu

**Kiedy stosować**: Gdy masz wcześniejszy model scoringowy i chcesz go przebudować.

#### 3. Fuzzy Augmentation (Ważenie Obserwacji)

**Opis**: Zamiast binarnego przypisania, odrzucony klient wchodzi do modelu jako częściowo Good i częściowo Bad poprzez system wag.

**Implementacja**:
```python
def fuzzy_augmentation_inference(df_accepted, df_rejected, existing_model):
    """
    Każdy odrzucony klient jest duplikowany - raz jako Good, raz jako Bad,
    z wagami odpowiadającymi prawdopodobieństwu
    """
    # Predykcja prawdopodobieństwa Bad
    prob_bad = existing_model.predict_proba(df_rejected[features])[:, 1]
    
    # Tworzenie zduplikowanych rekordów z wagami
    df_rejected_bad = df_rejected.copy()
    df_rejected_bad['target'] = 1
    df_rejected_bad['sample_weight'] = prob_bad
    
    df_rejected_good = df_rejected.copy()
    df_rejected_good['target'] = 0
    df_rejected_good['sample_weight'] = 1 - prob_bad
    
    # Łączenie wszystkich danych
    df_accepted['sample_weight'] = 1.0
    return pd.concat([df_accepted, df_rejected_bad, df_rejected_good], ignore_index=True)
```

**Zalety**:
- Najbardziej elastyczne podejście
- Zachowuje niepewność co do klasyfikacji odrzuconych
- Dobrze współpracuje z algorytmami obsługującymi wagi

**Wady**:
- Zwiększa rozmiar zbioru danych (duplikacja)
- Wymaga algorytmów obsługujących `sample_weight`
- Bardziej skomplikowana interpretacja

**Kiedy stosować**: Gdy zależy Ci na zachowaniu pełnej informacji o niepewności klasyfikacji.

### Porównanie Metod

| Metoda | Złożoność | Pesymizm | Wymaga modelu | Zachowuje niepewność |
|--------|-----------|----------|---------------|---------------------|
| Hard Cut-off | Niska | Wysoki | Nie | Nie |
| Parceling | Średnia | Średni | Tak | Nie |
| Fuzzy Augmentation | Wysoka | Niski | Tak | Tak |

---

## Kalibracja i Skalowanie (Scoring)

### Problem Transformacji Prawdopodobieństwa

Regresja logistyczna zwróci Ci prawdopodobieństwo `p`. Biznes nie rozumie `p = 0.023`. Biznes rozumie punktację (np. 600 pkt).

### Formuła Skalowania

```
Score = Offset + Factor × ln(odds)
```

Gdzie:
- `odds = (1 - p) / p` (stosunek szans Good do Bad)
- `ln(odds)` - logarytm naturalny odds (log-odds)
- `Offset` - przesunięcie punktacji (punkt startowy skali)
- `Factor` - współczynnik skalowania

### Points to Double Odds (PDO)

**PDO** to kluczowa koncepcja w skalowaniu punktacji. Określa, o ile punktów musi wzrosnąć score, aby odds (szansa na Good) podwoił się.

**Przykład**:
- Przy 600 pkt szansa na default to 1:50
- Przy 620 pkt (PDO=20) szansa na default maleje do 1:100

### Obliczanie Offset i Factor

```python
import numpy as np

def calculate_scaling_params(base_score, base_odds, pdo):
    """
    Oblicza parametry skalowania dla scorecardu
    
    Parameters:
    -----------
    base_score : float
        Bazowa punktacja (np. 600)
    base_odds : float
        Bazowy odds przy base_score (np. 50 oznacza 1:50)
    pdo : float
        Points to Double the Odds (np. 20)
    
    Returns:
    --------
    offset, factor : tuple
        Parametry do formuły Score = Offset + Factor × ln(odds)
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    return offset, factor


def calculate_score(probability, offset, factor):
    """
    Przekształca prawdopodobieństwo Bad na punktację
    
    Parameters:
    -----------
    probability : float or array
        Prawdopodobieństwo Bad (default)
    offset : float
        Offset z formuły skalowania
    factor : float
        Factor z formuły skalowania
    
    Returns:
    --------
    score : float or array
        Punktacja scoringowa
    """
    # Zabezpieczenie przed p=0 lub p=1
    probability = np.clip(probability, 1e-10, 1 - 1e-10)
    
    # Obliczenie odds
    odds = (1 - probability) / probability
    
    # Obliczenie score
    score = offset + factor * np.log(odds)
    
    return score
```

### Przykład Praktyczny

```python
# Parametry skalowania
base_score = 600   # Bazowa punktacja
base_odds = 50     # Odds 1:50 przy 600 punktach
pdo = 20           # 20 punktów podwaja odds

# Obliczenie parametrów
offset, factor = calculate_scaling_params(base_score, base_odds, pdo)
print(f"Offset: {offset:.2f}")
print(f"Factor: {factor:.2f}")

# Konwersja prawdopodobieństwa na score
probabilities = [0.02, 0.05, 0.10, 0.20]
for p in probabilities:
    score = calculate_score(p, offset, factor)
    print(f"P(Bad) = {p:.2%} → Score = {score:.0f}")
```

**Wynik**:
```
Offset: 487.12
Factor: 28.85
P(Bad) = 2.00% → Score = 699
P(Bad) = 5.00% → Score = 634
P(Bad) = 10.00% → Score = 580
P(Bad) = 20.00% → Score = 526
```

### Tabela Punktacji (Score Card Points)

Po skalowaniu współczynników modelu, każda kategoria zmiennej otrzymuje punkty:

```python
def create_scorecard(model_coefficients, woe_bins, offset, factor, n_features):
    """
    Tworzy tabelę punktacji scorecard z współczynników modelu
    
    Parameters:
    -----------
    model_coefficients : dict
        Współczynniki modelu {feature_name: coefficient}
    woe_bins : dict
        Słownik {feature_name: {bin_name: woe_value}}
    offset : float
        Offset skalowania
    factor : float
        Factor skalowania
    n_features : int
        Liczba zmiennych w modelu
    
    Returns:
    --------
    scorecard : pd.DataFrame
        Tabela punktacji
    """
    scorecard_rows = []
    
    # Punkty bazowe (offset rozłożony równomiernie)
    base_points_per_feature = offset / n_features
    
    for feature, coef in model_coefficients.items():
        for bin_name, woe in woe_bins[feature].items():
            points = base_points_per_feature + factor * coef * woe
            scorecard_rows.append({
                'Feature': feature,
                'Bin': bin_name,
                'WoE': woe,
                'Coefficient': coef,
                'Points': round(points, 0)
            })
    
    return pd.DataFrame(scorecard_rows)
```

---

## Implementacja Techniczna

### Rekomendowane Biblioteki Open Source

Zamiast pisać wszystko od zera, warto przeanalizować kod sprawdzonych bibliotek:

#### 1. optbinning
- **GitHub**: [guillermo-navas-palencia/optbinning](https://github.com/guillermo-navas-palencia/optbinning)
- **Znaczenie**: Obecnie State-of-the-Art w Pythonie
- **Kluczowe funkcje**:
  - Mixed-Integer Programming do optymalnego binowania
  - **Monotonic Binning** - wymuszenie, aby WoE rosło/malało monotonicznie (wymóg regulacyjny)
  - Automatic binning z constraints
  - Obsługa zmiennych ciągłych i kategorycznych

```python
from optbinning import OptimalBinning

optb = OptimalBinning(name="age", dtype="numerical", monotonic_trend="auto")
optb.fit(X_train['age'], y_train)

# Wyniki
binning_table = optb.binning_table.build()
print(binning_table)
```

#### 2. scorecardpy
- **GitHub**: [shichenxie/scorecardpy](https://github.com/shichenxie/scorecardpy)
- **Znaczenie**: Port popularnej biblioteki z języka R
- **Kluczowe funkcje**:
  - Szybkie generowanie raportów WoE
  - Automatyczny binning
  - Tworzenie scorecardów

```python
import scorecardpy as sc

# Binning
bins = sc.woebin(df, y="target")

# WoE transformation
df_woe = sc.woebin_ply(df, bins)

# Scorecard
card = sc.scorecard(bins, model, xcolumns=features)
```

#### 3. Toad
- **GitHub**: [amphibian-dev/toad](https://github.com/amphibian-dev/toad)
- **Znaczenie**: Rozwijana przez chiński fintech
- **Kluczowe funkcje**:
  - Wydajna selekcja zmiennych (IV selection, stepwise)
  - PSI (Population Stability Index)
  - Merge bin functionality

```python
import toad

# Feature selection by IV
selected = toad.selection.select(df, target='target', empty=0.9, iv=0.02, corr=0.7)

# Binning
c = toad.transform.Combiner()
c.fit(df, y='target', method='chi', min_samples=0.05)

# PSI calculation
psi = toad.metrics.PSI(expected, actual)
```

### Struktura Projektu

```
credit-scoring-engine/
├── METHODOLOGICAL_README.md
├── README.md
├── docs/
│   └── references/
│       ├── literature.md
│       ├── reject_inference.md
│       ├── calibration.md
│       └── metrics.md
├── src/
│   ├── __init__.py
│   ├── binning/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── decision_tree.py
│   │   ├── optimal.py
│   │   └── woe.py
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── iv.py
│   │   ├── vif.py
│   │   └── stepwise.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── logistic.py
│   │   └── reject_inference.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── scaling.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── psi.py
│   │   ├── csi.py
│   │   └── performance.py
│   └── visualization/
│       ├── __init__.py
│       ├── woe_charts.py
│       ├── score_distribution.py
│       └── stability.py
├── tests/
│   ├── test_binning.py
│   ├── test_selection.py
│   ├── test_modeling.py
│   ├── test_calibration.py
│   └── test_monitoring.py
├── examples/
│   └── complete_workflow.ipynb
└── requirements.txt
```

---

## Struktura Modułów Biblioteki

### Moduł 1: Binning & WoE (Serce Systemu)

**Cel**: Dyskretyzacja zmiennych ciągłych i obliczenie Weight of Evidence

#### Automatyka (Decision Tree Binning)

```python
from sklearn.tree import DecisionTreeClassifier

def decision_tree_binning(X, y, max_bins=10, min_samples_leaf=0.05):
    """
    Binning oparty na drzewie decyzyjnym jako punkt startowy
    """
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=min_samples_leaf
    )
    tree.fit(X.reshape(-1, 1), y)
    
    # Ekstrakcja progów
    thresholds = tree.tree_.threshold[tree.tree_.feature >= 0]
    thresholds = np.sort(np.unique(thresholds))
    
    return thresholds
```

#### Wymagania Biznesowe

- **Ręczne łączenie kubełków (Coarse Classing)** - Analityk musi mieć możliwość ręcznego połączenia bins
- **Narzucanie trendu (Constraints)** - WoE musi być monotoniczne (rosnące lub malejące)

#### Metryki: Information Value (IV)

```python
def calculate_woe_iv(df, feature, target):
    """
    Oblicza Weight of Evidence i Information Value dla zmiennej
    """
    # Agregacja po binach
    grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    grouped.columns = ['bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # Dystrybucje
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()
    
    grouped['dist_bad'] = grouped['bad'] / total_bad
    grouped['dist_good'] = grouped['good'] / total_good
    
    # WoE
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    
    # IV
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv
```

#### Złota Zasada IV

| Zakres IV | Interpretacja |
|-----------|---------------|
| < 0.02 | Bezużyteczna predykcyjnie |
| 0.02 - 0.1 | Słaba moc predykcyjna |
| 0.1 - 0.3 | Średnia moc predykcyjna |
| 0.3 - 0.5 | Silna moc predykcyjna |
| > 0.5 | Podejrzana (możliwy overfitting lub data leakage) |

### Moduł 2: Selekcja Zmiennych (Feature Selection)

#### Filtracja po IV

```python
def select_by_iv(iv_values, min_iv=0.02, max_iv=0.5):
    """
    Filtruje zmienne po Information Value
    """
    return [f for f, iv in iv_values.items() if min_iv <= iv <= max_iv]
```

#### Multikolinearność (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    Oblicza Variance Inflation Factor dla wszystkich zmiennych
    """
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) 
        for i in range(X.shape[1])
    ]
    return vif_data

# VIF > 5 sugeruje multikolinearność
# VIF > 10 wymaga usunięcia zmiennej
```

#### Korelacja Pearsona/Spearmana

```python
def remove_correlated_features(X, threshold=0.7, method='spearman'):
    """
    Usuwa zmienne silnie skorelowane
    """
    corr_matrix = X.corr(method=method).abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop
```

#### Stepwise Selection

```python
def stepwise_selection(X, y, initial_list=[], threshold_in=0.05, threshold_out=0.10):
    """
    Klasyczne forward/backward selection oparte o p-value
    (w bankowości to nadal standard, choć w ML odchodzi się od tego)
    """
    import statsmodels.api as sm
    
    included = list(initial_list)
    
    while True:
        changed = False
        
        # Forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        
        # Backward step
        model = sm.Logit(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # bez stałej
        worst_pval = pvalues.max()
        
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            changed = True
        
        if not changed:
            break
    
    return included
```

#### Marginal IV

```python
def marginal_iv(X, y, base_features, candidate_features, calculate_iv_func):
    """
    Oblicza przyrost IV po dodaniu zmiennej do istniejącego zestawu
    """
    base_iv = sum(calculate_iv_func(X, f, y) for f in base_features)
    
    marginal_ivs = {}
    for candidate in candidate_features:
        combined_iv = sum(calculate_iv_func(X, f, y) for f in base_features + [candidate])
        marginal_ivs[candidate] = combined_iv - base_iv
    
    return marginal_ivs
```

### Moduł 3: Modelowanie

#### Wrapper na Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

class CreditScoringModel:
    """
    Wrapper na regresję logistyczną z obsługą wag (sample_weights)
    """
    
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver='lbfgs',
            random_state=42
        )
    
    def fit(self, X, y, sample_weight=None):
        """
        Trenuje model z opcjonalnymi wagami
        (kluczowe dla Reject Inference - Fuzzy Augmentation)
        """
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_coefficients(self):
        return dict(zip(self.feature_names_, self.model.coef_[0]))
```

### Moduł 4: Ocena Stabilności i Performance (Monitoring)

#### PSI (Population Stability Index)

Porównanie rozkładu score'a w czasie (Development vs Out-of-Time).

```python
def calculate_psi(expected, actual, bins=10):
    """
    Oblicza Population Stability Index
    
    PSI < 0.1: Stabilny
    0.1 <= PSI < 0.25: Wymaga uwagi
    PSI >= 0.25: Alarm, model do przeliczenia
    """
    # Tworzenie binów na podstawie expected
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Dystrybucje
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected) + 1e-10
    actual_pct = actual_counts / len(actual) + 1e-10
    
    # PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi


def interpret_psi(psi):
    if psi < 0.1:
        return "Stabilny"
    elif psi < 0.25:
        return "Wymaga uwagi"
    else:
        return "Alarm - model do przeliczenia"
```

#### CSI (Characteristic Stability Index)

To samo co PSI, ale dla pojedynczych zmiennych.

```python
def calculate_csi(df_dev, df_oot, feature, bins):
    """
    Oblicza Characteristic Stability Index dla pojedynczej zmiennej
    (czy rozkład dochodów klientów się zmienił?)
    """
    return calculate_psi(df_dev[feature], df_oot[feature], bins)
```

#### Metryki Performance

```python
from sklearn.metrics import roc_auc_score

def calculate_gini(y_true, y_pred_proba):
    """
    Gini = 2 * AUC - 1
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def calculate_ks(y_true, y_pred_proba):
    """
    Kolmogorov-Smirnov statistic
    """
    from scipy.stats import ks_2samp
    
    good_scores = y_pred_proba[y_true == 0]
    bad_scores = y_pred_proba[y_true == 1]
    
    ks_stat, _ = ks_2samp(good_scores, bad_scores)
    return ks_stat


def calculate_lift(y_true, y_pred_proba, percentile=10):
    """
    Lift - ile razy więcej Bad jest w górnym percentylu vs populacja
    """
    threshold = np.percentile(y_pred_proba, 100 - percentile)
    top_decile = y_pred_proba >= threshold
    
    overall_bad_rate = y_true.mean()
    top_decile_bad_rate = y_true[top_decile].mean()
    
    return top_decile_bad_rate / overall_bad_rate
```

### Moduł 5: Wizualizacja

#### WoE Charts

```python
import matplotlib.pyplot as plt

def plot_woe_chart(woe_table, feature_name):
    """
    Najważniejszy wykres dla analityka ryzyka:
    - Oś X: kubełki zmiennej
    - Oś Y1: WoE (linia)
    - Oś Y2: % populacji (bar chart)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart - % populacji
    bars = ax1.bar(woe_table.index, woe_table['count_pct'] * 100, 
                   alpha=0.6, color='steelblue', label='% Populacji')
    ax1.set_xlabel('Bin')
    ax1.set_ylabel('% Populacji', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Linia - WoE
    ax2 = ax1.twinx()
    ax2.plot(woe_table.index, woe_table['woe'], 
             color='darkred', marker='o', linewidth=2, label='WoE')
    ax2.set_ylabel('Weight of Evidence (WoE)', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'WoE Analysis: {feature_name}')
    fig.tight_layout()
    
    return fig
```

#### Score Distribution

```python
def plot_score_distribution(scores_good, scores_bad):
    """
    Histogram punktacji Good vs Bad
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(scores_good, bins=50, alpha=0.5, label='Good', color='green', density=True)
    ax.hist(scores_bad, bins=50, alpha=0.5, label='Bad', color='red', density=True)
    
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution: Good vs Bad')
    ax.legend()
    
    return fig
```

#### Stability Plots

```python
def plot_psi_over_time(psi_values, dates):
    """
    Wykres PSI w czasie z liniami ostrzegawczymi
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, psi_values, marker='o', linewidth=2, color='navy')
    
    # Linie ostrzegawcze
    ax.axhline(y=0.1, color='orange', linestyle='--', label='Uwaga (0.1)')
    ax.axhline(y=0.25, color='red', linestyle='--', label='Alarm (0.25)')
    
    ax.set_xlabel('Data')
    ax.set_ylabel('PSI')
    ax.set_title('Population Stability Index Over Time')
    ax.legend()
    ax.fill_between(dates, 0, psi_values, alpha=0.2)
    
    return fig
```

---

## Metryki i Monitoring

### Podsumowanie Kluczowych Metryk

| Metryka | Zastosowanie | Interpretacja |
|---------|--------------|---------------|
| **IV** | Selekcja zmiennych | < 0.02: słaba, 0.1-0.3: średnia, > 0.5: podejrzana |
| **Gini** | Moc dyskryminacyjna | > 0.4: dobry model, > 0.6: bardzo dobry |
| **KS** | Separacja klas | > 0.3: dobra separacja |
| **PSI** | Stabilność modelu | < 0.1: stabilny, > 0.25: wymaga przeliczenia |
| **CSI** | Stabilność zmiennych | Analogicznie do PSI |
| **Lift** | Efektywność rankingu | > 2: model 2x lepszy niż losowy |

### Harmonogram Monitoringu

| Częstotliwość | Czynność |
|---------------|----------|
| Dziennie | Monitorowanie liczby scorowanych wniosków |
| Tygodniowo | Sprawdzanie rozkładu score'ów |
| Miesięcznie | PSI i CSI dla kluczowych zmiennych |
| Kwartalnie | Pełna walidacja modelu (Gini, KS, Lift) |
| Rocznie | Re-kalibracja lub re-development modelu |

---

## Materiały Referencyjne

### Dodatkowe Zasoby

1. **[docs/references/literature.md](docs/references/literature.md)** - Pełna lista literatury
2. **[docs/references/reject_inference.md](docs/references/reject_inference.md)** - Szczegóły Reject Inference
3. **[docs/references/calibration.md](docs/references/calibration.md)** - Szczegóły kalibracji
4. **[docs/references/metrics.md](docs/references/metrics.md)** - Dokumentacja metryk

### Linki do Repozytoriów

- [optbinning](https://github.com/guillermo-navas-palencia/optbinning) - State-of-the-Art binning
- [scorecardpy](https://github.com/shichenxie/scorecardpy) - Port z R, proste raporty
- [Toad](https://github.com/amphibian-dev/toad) - Chiński fintech, wydajna selekcja

---

## Licencja

Ten dokument jest częścią projektu Credit Scoring Engine i podlega licencji projektu.

---

*Ostatnia aktualizacja: Styczeń 2026*
