# Metryki i Monitoring Credit Scoringu

## Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Metryki Mocy Predykcyjnej](#metryki-mocy-predykcyjnej)
3. [Metryki Selekcji Zmiennych](#metryki-selekcji-zmiennych)
4. [Metryki Stabilności](#metryki-stabilności)
5. [Harmonogram Monitoringu](#harmonogram-monitoringu)
6. [Implementacja](#implementacja)

---

## Wprowadzenie

Monitoring modeli credit scoringowych jest krytyczny dla utrzymania ich skuteczności w czasie. Ten dokument opisuje kluczowe metryki używane w branży.

### Kategorie Metryk

| Kategoria | Cel | Przykłady |
|-----------|-----|-----------|
| Moc predykcyjna | Jak dobrze model rozróżnia Good od Bad | Gini, KS, AUC |
| Selekcja zmiennych | Które zmienne są najważniejsze | IV, WoE |
| Stabilność | Czy model jest stabilny w czasie | PSI, CSI |
| Kalibracja | Czy predykcje są dokładne | Brier Score, Calibration Plot |

---

## Metryki Mocy Predykcyjnej

### 1. Gini Coefficient

**Definicja**: Gini mierzy zdolność modelu do rozróżnienia między klasami. Jest powiązany z AUC (Area Under ROC Curve).

**Formuła**:
```
Gini = 2 × AUC - 1
```

**Interpretacja**:
| Gini | Interpretacja |
|------|---------------|
| < 0.20 | Słaby model |
| 0.20 - 0.40 | Akceptowalny model |
| 0.40 - 0.60 | Dobry model |
| 0.60 - 0.80 | Bardzo dobry model |
| > 0.80 | Doskonały (podejrzany - możliwy data leakage) |

**Implementacja**:
```python
from sklearn.metrics import roc_auc_score

def calculate_gini(y_true, y_pred_proba):
    """
    Oblicza współczynnik Gini
    
    Parameters:
    -----------
    y_true : array-like
        Rzeczywiste etykiety (0/1)
    y_pred_proba : array-like
        Przewidywane prawdopodobieństwo klasy 1 (Bad)
    
    Returns:
    --------
    gini : float
        Współczynnik Gini w zakresie [-1, 1]
    """
    auc = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc - 1
    return gini
```

---

### 2. KS (Kolmogorov-Smirnov) Statistic

**Definicja**: Maksymalna różnica między skumulowanymi dystrybuantami klas Good i Bad.

**Formuła**:
```
KS = max|F_good(x) - F_bad(x)|
```

**Interpretacja**:
| KS | Interpretacja |
|----|---------------|
| < 0.20 | Słaby model |
| 0.20 - 0.40 | Akceptowalny model |
| 0.40 - 0.60 | Dobry model |
| 0.60 - 0.75 | Bardzo dobry model |
| > 0.75 | Doskonały (podejrzany) |

**Implementacja**:
```python
import numpy as np

def calculate_ks(y_true, y_pred_proba, n_bins=100):
    """
    Oblicza statystykę Kolmogorov-Smirnov
    
    Parameters:
    -----------
    y_true : array-like
        Rzeczywiste etykiety (0/1)
    y_pred_proba : array-like
        Przewidywane prawdopodobieństwo klasy 1 (Bad)
    n_bins : int
        Liczba punktów do obliczenia KS
    
    Returns:
    --------
    ks_stat : float
        Statystyka KS
    ks_threshold : float
        Wartość progowa przy której KS jest maksymalne
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Sortowanie po prawdopodobieństwie
    sorted_indices = np.argsort(y_pred_proba)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred_proba[sorted_indices]
    
    # Obliczenie skumulowanych dystrybucji
    n_good = (y_true == 0).sum()
    n_bad = (y_true == 1).sum()
    
    cum_good = np.cumsum(y_true_sorted == 0) / n_good
    cum_bad = np.cumsum(y_true_sorted == 1) / n_bad
    
    # KS jako maksymalna różnica
    ks_values = np.abs(cum_good - cum_bad)
    ks_stat = np.max(ks_values)
    ks_idx = np.argmax(ks_values)
    ks_threshold = y_pred_sorted[ks_idx]
    
    return ks_stat, ks_threshold


def ks_curve(y_true, y_pred_proba):
    """
    Generuje dane do wykresu KS
    """
    sorted_indices = np.argsort(y_pred_proba)
    y_true_sorted = y_true[sorted_indices]
    
    n_good = (y_true == 0).sum()
    n_bad = (y_true == 1).sum()
    
    cum_good = np.cumsum(y_true_sorted == 0) / n_good
    cum_bad = np.cumsum(y_true_sorted == 1) / n_bad
    
    return cum_good, cum_bad
```

---

### 3. Lift

**Definicja**: O ile razy więcej "Bad" jest w górnym percentylu w porównaniu do całej populacji.

**Formuła**:
```
Lift(top p%) = Bad Rate w top p% / Overall Bad Rate
```

**Interpretacja**:
- Lift > 1 oznacza, że model jest lepszy niż losowy wybór
- Lift = 2 w top 10% oznacza, że model identyfikuje 2x więcej Bad niż losowo

**Implementacja**:
```python
def calculate_lift(y_true, y_pred_proba, percentile=10):
    """
    Oblicza Lift dla danego percentyla
    
    Parameters:
    -----------
    y_true : array-like
        Rzeczywiste etykiety (0/1)
    y_pred_proba : array-like
        Przewidywane prawdopodobieństwo klasy 1 (Bad)
    percentile : float
        Percentyl do analizy (np. 10 = top 10%)
    
    Returns:
    --------
    lift : float
        Wartość Lift
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Próg dla danego percentyla (najwyższe prawdopodobieństwa)
    threshold = np.percentile(y_pred_proba, 100 - percentile)
    
    # Wybór top percentyla
    top_mask = y_pred_proba >= threshold
    
    # Obliczenie Bad Rate
    overall_bad_rate = y_true.mean()
    top_bad_rate = y_true[top_mask].mean()
    
    lift = top_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0
    
    return lift


def lift_curve(y_true, y_pred_proba, n_bins=10):
    """
    Generuje krzywą Lift po decylach
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Sortowanie malejąco
    sorted_indices = np.argsort(-y_pred_proba)
    y_true_sorted = y_true[sorted_indices]
    
    overall_bad_rate = y_true.mean()
    
    deciles = []
    lifts = []
    cumulative_lifts = []
    
    n = len(y_true)
    decile_size = n // n_bins
    
    for i in range(n_bins):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < n_bins - 1 else n
        
        decile_bad_rate = y_true_sorted[start_idx:end_idx].mean()
        cumulative_bad_rate = y_true_sorted[:end_idx].mean()
        
        deciles.append(i + 1)
        lifts.append(decile_bad_rate / overall_bad_rate)
        cumulative_lifts.append(cumulative_bad_rate / overall_bad_rate)
    
    return deciles, lifts, cumulative_lifts
```

---

### 4. Cumulative Accuracy Profile (CAP)

**Definicja**: Krzywa pokazująca skumulowany procent Bad uchwycony przez model.

**Implementacja**:
```python
def cap_curve(y_true, y_pred_proba):
    """
    Generuje krzywą CAP (Cumulative Accuracy Profile)
    """
    sorted_indices = np.argsort(-y_pred_proba)
    y_true_sorted = y_true[sorted_indices]
    
    n = len(y_true)
    n_bad = y_true.sum()
    
    x_axis = np.arange(1, n + 1) / n  # % populacji
    y_axis = np.cumsum(y_true_sorted) / n_bad  # % Bad uchwyconych
    
    return x_axis, y_axis
```

---

## Metryki Selekcji Zmiennych

### 1. Information Value (IV)

**Definicja**: Miara mocy predykcyjnej zmiennej.

**Formuła**:
```
IV = Σ (Dist_Good_i - Dist_Bad_i) × WoE_i
```

Gdzie:
```
WoE_i = ln(Dist_Good_i / Dist_Bad_i)
Dist_Good_i = % Good w binie i
Dist_Bad_i = % Bad w binie i
```

**Interpretacja**:
| IV | Moc predykcyjna |
|----|-----------------|
| < 0.02 | Bezużyteczna |
| 0.02 - 0.1 | Słaba |
| 0.1 - 0.3 | Średnia |
| 0.3 - 0.5 | Silna |
| > 0.5 | Podejrzana (możliwy overfitting) |

**Implementacja**:
```python
def calculate_woe_iv(df, feature, target, bins=None):
    """
    Oblicza Weight of Evidence i Information Value
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame z danymi
    feature : str
        Nazwa zmiennej
    target : str
        Nazwa zmiennej target (0/1)
    bins : int or array-like, optional
        Liczba binów lub progi binowania
    
    Returns:
    --------
    woe_table : pd.DataFrame
        Tabela z WoE dla każdego binu
    iv : float
        Information Value
    """
    df_temp = df[[feature, target]].copy()
    
    # Binowanie dla zmiennych ciągłych
    if bins is not None and df[feature].dtype in ['float64', 'int64']:
        df_temp[feature] = pd.cut(df_temp[feature], bins=bins, duplicates='drop')
    
    # Agregacja
    grouped = df_temp.groupby(feature)[target].agg(['sum', 'count'])
    grouped.columns = ['bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # Zabezpieczenie przed zerami
    grouped['bad'] = grouped['bad'].replace(0, 0.5)
    grouped['good'] = grouped['good'].replace(0, 0.5)
    
    # Dystrybucje
    total_bad = grouped['bad'].sum()
    total_good = grouped['good'].sum()
    
    grouped['dist_bad'] = grouped['bad'] / total_bad
    grouped['dist_good'] = grouped['good'] / total_good
    
    # WoE i IV
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    
    return grouped, total_iv


def calculate_all_iv(df, features, target, bins=10):
    """
    Oblicza IV dla wszystkich zmiennych
    """
    iv_values = {}
    for feature in features:
        _, iv = calculate_woe_iv(df, feature, target, bins=bins)
        iv_values[feature] = iv
    
    return pd.Series(iv_values).sort_values(ascending=False)
```

---

### 2. VIF (Variance Inflation Factor)

**Definicja**: Miara multikolinearności między zmiennymi.

**Formuła**:
```
VIF_i = 1 / (1 - R²_i)
```

Gdzie R²_i to R-kwadrat regresji zmiennej i na wszystkich innych zmiennych.

**Interpretacja**:
| VIF | Interpretacja |
|-----|---------------|
| 1 | Brak korelacji |
| 1 - 5 | Umiarkowana korelacja |
| 5 - 10 | Wysoka korelacja |
| > 10 | Bardzo wysoka - rozważ usunięcie |

**Implementacja**:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    Oblicza VIF dla wszystkich zmiennych
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame ze zmiennymi numerycznymi
    
    Returns:
    --------
    vif_data : pd.DataFrame
        Tabela z VIF dla każdej zmiennej
    """
    X_array = X.values
    
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_array, i) 
        for i in range(X_array.shape[1])
    ]
    
    return vif_data.sort_values('VIF', ascending=False)


def remove_high_vif(X, threshold=5.0):
    """
    Iteracyjnie usuwa zmienne z wysokim VIF
    """
    X_temp = X.copy()
    removed = []
    
    while True:
        vif = calculate_vif(X_temp)
        max_vif = vif['VIF'].max()
        
        if max_vif > threshold:
            worst_feature = vif.loc[vif['VIF'].idxmax(), 'feature']
            X_temp = X_temp.drop(columns=[worst_feature])
            removed.append((worst_feature, max_vif))
            print(f"Usunięto {worst_feature} (VIF={max_vif:.2f})")
        else:
            break
    
    return X_temp, removed
```

---

## Metryki Stabilności

### 1. PSI (Population Stability Index)

**Definicja**: Mierzy zmianę rozkładu score'a między dwoma okresami (np. development vs production).

**Formuła**:
```
PSI = Σ (Actual_i - Expected_i) × ln(Actual_i / Expected_i)
```

**Interpretacja**:
| PSI | Interpretacja | Akcja |
|-----|---------------|-------|
| < 0.1 | Stabilny | Brak |
| 0.1 - 0.25 | Umiarkowana zmiana | Monitoruj |
| ≥ 0.25 | Znacząca zmiana | Przebuduj model |

**Implementacja**:
```python
def calculate_psi(expected, actual, bins=10, epsilon=1e-4):
    """
    Oblicza Population Stability Index
    
    Parameters:
    -----------
    expected : array-like
        Rozkład bazowy (development)
    actual : array-like
        Rozkład aktualny (production)
    bins : int or array-like
        Liczba binów lub progi binowania
    epsilon : float
        Mała wartość dodawana do uniknięcia dzielenia przez zero
    
    Returns:
    --------
    psi : float
        Population Stability Index
    psi_table : pd.DataFrame
        Tabela z PSI dla każdego binu
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Tworzenie binów na podstawie expected
    if isinstance(bins, int):
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
    else:
        breakpoints = bins
    
    # Dystrybucje
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected) + epsilon
    actual_pct = actual_counts / len(actual) + epsilon
    
    # PSI per bin
    psi_per_bin = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    
    # Total PSI
    psi = np.sum(psi_per_bin)
    
    # Tabela
    psi_table = pd.DataFrame({
        'Bin': range(1, len(psi_per_bin) + 1),
        'Expected_Pct': expected_pct - epsilon,
        'Actual_Pct': actual_pct - epsilon,
        'PSI_Contribution': psi_per_bin
    })
    
    return psi, psi_table


def interpret_psi(psi):
    """Interpretuje wartość PSI"""
    if psi < 0.1:
        return "Stabilny", "green"
    elif psi < 0.25:
        return "Umiarkowana zmiana", "orange"
    else:
        return "Znacząca zmiana - wymaga przebudowy", "red"
```

---

### 2. CSI (Characteristic Stability Index)

**Definicja**: Jak PSI, ale dla pojedynczych zmiennych (nie score'a).

**Implementacja**:
```python
def calculate_csi_all_features(df_expected, df_actual, features, bins=10):
    """
    Oblicza CSI dla wszystkich zmiennych
    
    Parameters:
    -----------
    df_expected : pd.DataFrame
        Dane bazowe (development)
    df_actual : pd.DataFrame
        Dane aktualne (production)
    features : list
        Lista zmiennych do analizy
    bins : int
        Liczba binów
    
    Returns:
    --------
    csi_summary : pd.DataFrame
        Podsumowanie CSI dla wszystkich zmiennych
    """
    results = []
    
    for feature in features:
        csi, _ = calculate_psi(
            df_expected[feature].dropna(),
            df_actual[feature].dropna(),
            bins=bins
        )
        interpretation, color = interpret_psi(csi)
        
        results.append({
            'Feature': feature,
            'CSI': csi,
            'Interpretation': interpretation
        })
    
    return pd.DataFrame(results).sort_values('CSI', ascending=False)
```

---

### 3. Herfindahl-Hirschman Index (HHI)

**Definicja**: Mierzy koncentrację score'ów w określonych binach.

**Interpretacja**: Wysoki HHI oznacza, że większość obserwacji ma podobny score (mała dyskryminacja).

**Implementacja**:
```python
def calculate_hhi(scores, n_bins=10):
    """
    Oblicza Herfindahl-Hirschman Index
    """
    bins = pd.qcut(scores, n_bins, duplicates='drop')
    proportions = bins.value_counts(normalize=True)
    hhi = (proportions ** 2).sum()
    return hhi
```

---

## Harmonogram Monitoringu

### Częstotliwość Monitoringu

| Częstotliwość | Metryki | Cel |
|---------------|---------|-----|
| Dziennie | Liczba wniosków, średni score | Wykrywanie anomalii |
| Tygodniowo | Rozkład score'ów, progi akceptacji | Trend krótkoterminowy |
| Miesięcznie | PSI, CSI kluczowych zmiennych | Stabilność |
| Kwartalnie | Gini, KS, Lift (na danych z wynikiem) | Performance |
| Rocznie | Pełna re-walidacja | Re-development jeśli potrzeba |

### Dashboard Monitoringu

```python
class ScoringModelMonitor:
    """
    Klasa do kompleksowego monitoringu modelu scoringowego
    """
    
    def __init__(self, model, calibrator, baseline_data, baseline_scores):
        self.model = model
        self.calibrator = calibrator
        self.baseline_data = baseline_data
        self.baseline_scores = baseline_scores
    
    def generate_report(self, current_data, current_scores, current_target=None):
        """
        Generuje raport monitoringu
        """
        report = {}
        
        # 1. PSI Score
        psi, psi_table = calculate_psi(self.baseline_scores, current_scores)
        report['psi'] = psi
        report['psi_interpretation'] = interpret_psi(psi)[0]
        
        # 2. Statystyki opisowe
        report['score_stats'] = {
            'baseline_mean': self.baseline_scores.mean(),
            'current_mean': current_scores.mean(),
            'baseline_std': self.baseline_scores.std(),
            'current_std': current_scores.std()
        }
        
        # 3. Performance (jeśli mamy target)
        if current_target is not None:
            probs = self.model.predict_proba(current_data)[:, 1]
            report['gini'] = calculate_gini(current_target, probs)
            report['ks'], _ = calculate_ks(current_target, probs)
            report['lift_top10'] = calculate_lift(current_target, probs, 10)
        
        return report
    
    def alert_check(self, report, psi_threshold=0.25, gini_min=0.30):
        """
        Sprawdza czy potrzebne są alerty
        """
        alerts = []
        
        if report['psi'] >= psi_threshold:
            alerts.append(f"ALERT: PSI = {report['psi']:.3f} >= {psi_threshold}")
        
        if 'gini' in report and report['gini'] < gini_min:
            alerts.append(f"ALERT: Gini = {report['gini']:.3f} < {gini_min}")
        
        return alerts
```

---

## Implementacja

### Kompletna Klasa Metryk

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

class CreditScoringMetrics:
    """
    Kompletny zestaw metryk dla credit scoringu
    """
    
    @staticmethod
    def gini(y_true, y_pred_proba):
        return 2 * roc_auc_score(y_true, y_pred_proba) - 1
    
    @staticmethod
    def ks(y_true, y_pred_proba):
        ks_stat, _ = calculate_ks(y_true, y_pred_proba)
        return ks_stat
    
    @staticmethod
    def lift(y_true, y_pred_proba, percentile=10):
        return calculate_lift(y_true, y_pred_proba, percentile)
    
    @staticmethod
    def psi(expected, actual, bins=10):
        psi_value, _ = calculate_psi(expected, actual, bins)
        return psi_value
    
    @staticmethod
    def brier(y_true, y_pred_proba):
        return brier_score_loss(y_true, y_pred_proba)
    
    @staticmethod
    def summary(y_true, y_pred_proba):
        """Generuje podsumowanie wszystkich metryk"""
        return {
            'Gini': CreditScoringMetrics.gini(y_true, y_pred_proba),
            'KS': CreditScoringMetrics.ks(y_true, y_pred_proba),
            'Lift@10%': CreditScoringMetrics.lift(y_true, y_pred_proba, 10),
            'Lift@20%': CreditScoringMetrics.lift(y_true, y_pred_proba, 20),
            'Brier': CreditScoringMetrics.brier(y_true, y_pred_proba)
        }
```

---

## Podsumowanie

| Metryka | Zastosowanie | Wartość Docelowa |
|---------|--------------|------------------|
| Gini | Dyskryminacja | > 0.40 |
| KS | Separacja | > 0.30 |
| Lift@10% | Efektywność | > 2.0 |
| IV | Moc zmiennej | 0.1 - 0.5 |
| VIF | Kolinearność | < 5.0 |
| PSI | Stabilność | < 0.10 |

Regularne monitorowanie tych metryk pozwala utrzymać wysoką jakość modelu scoringowego i reagować na zmiany w populacji klientów.

---

*Ostatnia aktualizacja: Styczeń 2026*
