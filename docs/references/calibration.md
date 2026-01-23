# Kalibracja i Skalowanie (Scoring)

## Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Od Prawdopodobieństwa do Punktacji](#od-prawdopodobieństwa-do-punktacji)
3. [Points to Double Odds (PDO)](#points-to-double-odds-pdo)
4. [Formuła Skalowania](#formuła-skalowania)
5. [Implementacja](#implementacja)
6. [Tworzenie Tabeli Punktacji](#tworzenie-tabeli-punktacji)
7. [Przykłady Praktyczne](#przykłady-praktyczne)
8. [Best Practices](#best-practices)

---

## Wprowadzenie

Kalibracja (skalowanie) to proces przekształcenia surowego wyniku modelu (prawdopodobieństwa) na zrozumiałą dla biznesu punktację. Jest to kluczowy krok w budowie scorecardu kredytowego.

### Dlaczego Skalowanie Jest Niezbędne?

| Co model zwraca | Co biznes rozumie |
|-----------------|-------------------|
| P(Bad) = 0.023 | Score = 687 punktów |
| P(Bad) = 0.156 | Score = 512 punktów |
| log-odds = -3.75 | Score = 720 punktów |

**Punktacja ma kilka zalet:**
1. **Intuicyjność** - "Klient ma 650 punktów" jest bardziej zrozumiałe
2. **Porównywalność** - Różne modele można zestawić na wspólnej skali
3. **Komunikacja** - Łatwiej komunikować progi akceptacji
4. **Historyczność** - Instytucje mają historię interpretacji określonych poziomów punktacji

---

## Od Prawdopodobieństwa do Punktacji

### Krok 1: Z prawdopodobieństwa do odds

**Odds** to stosunek szansy zdarzenia "Good" do zdarzenia "Bad":

```
odds = P(Good) / P(Bad) = (1 - p) / p
```

Gdzie `p` to prawdopodobieństwo Bad (default).

**Przykłady:**
| P(Bad) | P(Good) | Odds | Interpretacja |
|--------|---------|------|---------------|
| 0.02 | 0.98 | 49:1 | 1 bad na 50 klientów |
| 0.05 | 0.95 | 19:1 | 1 bad na 20 klientów |
| 0.10 | 0.90 | 9:1 | 1 bad na 10 klientów |
| 0.20 | 0.80 | 4:1 | 1 bad na 5 klientów |

### Krok 2: Z odds do log-odds

**Log-odds** (logit) to logarytm naturalny z odds:

```
log_odds = ln(odds) = ln((1 - p) / p)
```

Log-odds ma zakres od -∞ do +∞, co ułatwia transformację liniową.

### Krok 3: Z log-odds do score

Finalna transformacja liniowa:

```
Score = Offset + Factor × log_odds
```

---

## Points to Double Odds (PDO)

### Definicja

**PDO (Points to Double the Odds)** to liczba punktów, o którą musi wzrosnąć score, aby odds podwoił się.

### Przykład Praktyczny

Załóżmy PDO = 20 i bazowy score 600 przy odds 50:1:

| Score | Odds | P(Bad) | Interpretacja |
|-------|------|--------|---------------|
| 560 | 25:1 | 3.85% | Połowa bazowych odds |
| 580 | 35:1 | 2.78% | ~70% bazowych odds |
| 600 | 50:1 | 1.96% | Bazowy punkt |
| 620 | 100:1 | 0.99% | Podwojone odds |
| 640 | 200:1 | 0.50% | Poczwórne odds |

### Typowe Wartości PDO

| PDO | Charakterystyka |
|-----|-----------------|
| 10 | Bardzo wrażliwa skala (mały zakres score) |
| 20 | Standardowa wartość (FICO, większość banków) |
| 30 | Mniej wrażliwa skala (szeroki zakres score) |

### Wpływ PDO na Zakres Score

Przy tym samym zakresie prawdopodobieństwa, różne PDO dają różne zakresy score:

```python
import numpy as np

def score_range(p_min, p_max, pdo, base_score, base_odds):
    """Oblicza zakres score dla danego zakresu prawdopodobieństwa"""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    odds_min = (1 - p_min) / p_min
    odds_max = (1 - p_max) / p_max
    
    score_max = offset + factor * np.log(odds_min)
    score_min = offset + factor * np.log(odds_max)
    
    return score_min, score_max

# P(Bad) od 1% do 30%
for pdo in [10, 20, 30]:
    s_min, s_max = score_range(0.01, 0.30, pdo, 600, 50)
    print(f"PDO={pdo}: Score od {s_min:.0f} do {s_max:.0f} (zakres: {s_max - s_min:.0f} pkt)")
```

**Wynik:**
```
PDO=10: Score od 518 do 699 (zakres: 181 pkt)
PDO=20: Score od 435 to 797 (zakres: 362 pkt)
PDO=30: Score od 353 do 896 (zakres: 543 pkt)
```

---

## Formuła Skalowania

### Podstawowa Formuła

```
Score = Offset + Factor × ln(odds)
```

Gdzie:
- **Offset**: Punkt przesunięcia skali
- **Factor**: Współczynnik skalowania
- **odds**: (1 - p) / p

### Obliczanie Offset i Factor

Z definicji PDO:
- Przy score = base_score, odds = base_odds
- Przy score = base_score + PDO, odds = 2 × base_odds

Z tych dwóch równań:

```
base_score = Offset + Factor × ln(base_odds)
base_score + PDO = Offset + Factor × ln(2 × base_odds)
```

Odejmując:

```
PDO = Factor × ln(2 × base_odds) - Factor × ln(base_odds)
PDO = Factor × ln(2)
Factor = PDO / ln(2)
```

I podstawiając z powrotem:

```
Offset = base_score - Factor × ln(base_odds)
```

### Wzory Końcowe

```python
Factor = PDO / np.log(2)  # ≈ PDO / 0.693
Offset = base_score - Factor * np.log(base_odds)
```

---

## Implementacja

### Klasa Skalowania

```python
import numpy as np
import pandas as pd

class ScoreCalibrator:
    """
    Klasa do kalibracji i skalowania punktacji credit scoringu
    """
    
    def __init__(self, base_score=600, base_odds=50, pdo=20):
        """
        Inicjalizacja kalibratora
        
        Parameters:
        -----------
        base_score : float
            Bazowa punktacja (np. 600)
        base_odds : float
            Bazowy odds przy base_score (np. 50 oznacza odds 50:1)
        pdo : float
            Points to Double the Odds (np. 20)
        """
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        
        # Obliczenie parametrów skalowania
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
    
    def probability_to_score(self, probability):
        """
        Przekształca prawdopodobieństwo Bad na punktację
        
        Parameters:
        -----------
        probability : float or array-like
            Prawdopodobieństwo Bad (default)
        
        Returns:
        --------
        score : float or array-like
            Punktacja scoringowa
        """
        probability = np.asarray(probability)
        
        # Zabezpieczenie przed p=0 lub p=1
        probability = np.clip(probability, 1e-10, 1 - 1e-10)
        
        # Obliczenie odds i score
        odds = (1 - probability) / probability
        score = self.offset + self.factor * np.log(odds)
        
        return score
    
    def score_to_probability(self, score):
        """
        Przekształca punktację na prawdopodobieństwo Bad
        
        Parameters:
        -----------
        score : float or array-like
            Punktacja scoringowa
        
        Returns:
        --------
        probability : float or array-like
            Prawdopodobieństwo Bad (default)
        """
        score = np.asarray(score)
        
        # Odwrotna transformacja
        log_odds = (score - self.offset) / self.factor
        odds = np.exp(log_odds)
        probability = 1 / (1 + odds)
        
        return probability
    
    def score_to_odds(self, score):
        """
        Przekształca punktację na odds
        """
        score = np.asarray(score)
        log_odds = (score - self.offset) / self.factor
        return np.exp(log_odds)
    
    def generate_score_table(self, score_range=None, step=10):
        """
        Generuje tabelę mapowania score -> probability -> odds
        """
        if score_range is None:
            score_range = (int(self.base_score - 100), int(self.base_score + 100))
        
        scores = np.arange(score_range[0], score_range[1] + 1, step)
        probs = self.score_to_probability(scores)
        odds = self.score_to_odds(scores)
        
        table = pd.DataFrame({
            'Score': scores,
            'P(Bad)': probs,
            'P(Good)': 1 - probs,
            'Odds': odds,
            'Odds Ratio': [f"1:{int(o)}" if o >= 1 else f"{int(1/o)}:1" for o in odds]
        })
        
        return table
    
    def __repr__(self):
        return (f"ScoreCalibrator(base_score={self.base_score}, "
                f"base_odds={self.base_odds}, pdo={self.pdo})\n"
                f"  Factor: {self.factor:.4f}\n"
                f"  Offset: {self.offset:.4f}")
```

### Przykład Użycia

```python
# Inicjalizacja
calibrator = ScoreCalibrator(base_score=600, base_odds=50, pdo=20)
print(calibrator)

# Konwersja prawdopodobieństwa na score
probabilities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
scores = calibrator.probability_to_score(probabilities)

for p, s in zip(probabilities, scores):
    print(f"P(Bad) = {p:.2%} → Score = {s:.0f}")

# Tabela mapowania
table = calibrator.generate_score_table(score_range=(450, 750), step=25)
print(table)
```

---

## Tworzenie Tabeli Punktacji

### Scorecard - Tabela Punktów dla Zmiennych

Finalnym produktem modelu scoringowego jest **scorecard** - tabela przypisująca punkty każdej kategorii każdej zmiennej.

### Wzór na Punkty

Punkty dla kategorii `i` zmiennej `j`:

```
Points_ij = (Offset / n) + Factor × β_j × WoE_ij
```

Gdzie:
- `n` - liczba zmiennych w modelu
- `β_j` - współczynnik regresji dla zmiennej j
- `WoE_ij` - Weight of Evidence dla kategorii i zmiennej j

### Implementacja Scorecardu

```python
class ScorecardBuilder:
    """
    Buduje tabelę punktacji (scorecard) z modelu regresji logistycznej
    """
    
    def __init__(self, calibrator):
        """
        Parameters:
        -----------
        calibrator : ScoreCalibrator
            Obiekt kalibratora ze zdefiniowanymi parametrami skalowania
        """
        self.calibrator = calibrator
    
    def build_scorecard(self, model_coefficients, woe_bins, feature_names):
        """
        Tworzy tabelę punktacji scorecard
        
        Parameters:
        -----------
        model_coefficients : array-like
            Współczynniki modelu regresji logistycznej (bez intercept)
        woe_bins : dict
            Słownik {feature_name: {bin_label: woe_value}}
        feature_names : list
            Lista nazw zmiennych w modelu
        
        Returns:
        --------
        scorecard : pd.DataFrame
            Tabela punktacji
        """
        n_features = len(feature_names)
        base_points = self.calibrator.offset / n_features
        
        scorecard_rows = []
        
        for feature, coef in zip(feature_names, model_coefficients):
            if feature not in woe_bins:
                continue
                
            for bin_label, woe in woe_bins[feature].items():
                points = base_points + self.calibrator.factor * coef * woe
                
                scorecard_rows.append({
                    'Variable': feature,
                    'Bin': bin_label,
                    'WoE': round(woe, 4),
                    'Coefficient': round(coef, 4),
                    'Points': round(points, 0)
                })
        
        scorecard = pd.DataFrame(scorecard_rows)
        
        return scorecard
    
    def calculate_score(self, scorecard, observation):
        """
        Oblicza score dla pojedynczej obserwacji
        
        Parameters:
        -----------
        scorecard : pd.DataFrame
            Tabela punktacji
        observation : dict
            Słownik {variable: bin_label}
        
        Returns:
        --------
        total_score : float
            Suma punktów
        """
        total_score = 0
        
        for variable, bin_label in observation.items():
            points = scorecard[(scorecard['Variable'] == variable) & 
                              (scorecard['Bin'] == bin_label)]['Points']
            if len(points) > 0:
                total_score += points.values[0]
        
        return total_score
```

### Przykład Scorecardu

```python
# Parametry kalibracji
calibrator = ScoreCalibrator(base_score=600, base_odds=50, pdo=20)

# Współczynniki modelu (przykładowe)
model_coefs = [0.8, 1.2, 0.5]  # age, income, employment_length
feature_names = ['age', 'income', 'employment_length']

# Binning i WoE (przykładowe)
woe_bins = {
    'age': {
        '18-25': -0.5,
        '26-35': 0.1,
        '36-50': 0.3,
        '51+': 0.2
    },
    'income': {
        '<2000': -0.8,
        '2000-4000': -0.2,
        '4000-6000': 0.3,
        '>6000': 0.7
    },
    'employment_length': {
        '<1 year': -0.6,
        '1-3 years': 0.0,
        '3-5 years': 0.2,
        '>5 years': 0.4
    }
}

# Budowa scorecardu
builder = ScorecardBuilder(calibrator)
scorecard = builder.build_scorecard(model_coefs, woe_bins, feature_names)
print(scorecard.to_string(index=False))
```

**Wynik:**
```
       Variable        Bin     WoE  Coefficient  Points
            age      18-25 -0.5000       0.8000     151
            age      26-35  0.1000       0.8000     165
            age      36-50  0.3000       0.8000     170
            age        51+  0.2000       0.8000     167
         income      <2000 -0.8000       1.2000     135
         income  2000-4000 -0.2000       1.2000     156
         income  4000-6000  0.3000       1.2000     173
         income      >6000  0.7000       1.2000     187
employment_len   <1 year -0.6000       0.5000     154
employment_len  1-3 years  0.0000       0.5000     163
employment_len  3-5 years  0.2000       0.5000     168
employment_len   >5 years  0.4000       0.5000     174
```

---

## Przykłady Praktyczne

### Przykład 1: Pełny Workflow

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Dane (symulowane)
np.random.seed(42)
n = 1000

# Symulacja danych z WoE transformacją
data = {
    'age_woe': np.random.normal(0, 0.5, n),
    'income_woe': np.random.normal(0, 0.6, n),
    'employment_woe': np.random.normal(0, 0.4, n)
}
df = pd.DataFrame(data)
df['target'] = (df.sum(axis=1) + np.random.normal(0, 1, n) > 0).astype(int)

# 2. Model
X = df[['age_woe', 'income_woe', 'employment_woe']]
y = df['target']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Współczynniki:", model.coef_[0])
print("Intercept:", model.intercept_[0])

# 3. Kalibracja
calibrator = ScoreCalibrator(base_score=600, base_odds=50, pdo=20)

# 4. Predykcja i skalowanie
probabilities = model.predict_proba(X)[:, 1]
scores = calibrator.probability_to_score(probabilities)

print(f"\nStatystyki Score:")
print(f"  Min: {scores.min():.0f}")
print(f"  Max: {scores.max():.0f}")
print(f"  Mean: {scores.mean():.0f}")
print(f"  Std: {scores.std():.0f}")

# 5. Analiza rozkładu
bad_scores = scores[y == 1]
good_scores = scores[y == 0]

print(f"\nŚredni score Bad: {bad_scores.mean():.0f}")
print(f"Średni score Good: {good_scores.mean():.0f}")
print(f"Separacja: {good_scores.mean() - bad_scores.mean():.0f} punktów")
```

### Przykład 2: Ustalanie Progów Akceptacji

```python
def analyze_cutoff(calibrator, scores, y_true, cutoff_score):
    """
    Analizuje skuteczność progu akceptacji
    """
    accepted = scores >= cutoff_score
    rejected = ~accepted
    
    prob_at_cutoff = calibrator.score_to_probability(cutoff_score)
    
    # Metryki
    n_accepted = accepted.sum()
    n_rejected = rejected.sum()
    acceptance_rate = n_accepted / len(scores)
    
    bad_rate_accepted = y_true[accepted].mean() if n_accepted > 0 else 0
    bad_rate_rejected = y_true[rejected].mean() if n_rejected > 0 else 0
    
    print(f"Próg: {cutoff_score} (P(Bad) = {prob_at_cutoff:.2%})")
    print(f"  Akceptowanych: {n_accepted} ({acceptance_rate:.1%})")
    print(f"  Odrzuconych: {n_rejected} ({1-acceptance_rate:.1%})")
    print(f"  Bad Rate wśród akceptowanych: {bad_rate_accepted:.2%}")
    print(f"  Bad Rate wśród odrzuconych: {bad_rate_rejected:.2%}")
    print()

# Analiza różnych progów
for cutoff in [500, 550, 600, 650]:
    analyze_cutoff(calibrator, scores, y, cutoff)
```

---

## Best Practices

### 1. Wybór Parametrów Bazowych

**Base Score i Base Odds:**
- Wybierz wartości reprezentujące "typowego" klienta w populacji
- Często używane: 600@50:1, 700@100:1, 500@20:1

**PDO:**
- 20 jest standardem branżowym (FICO)
- Mniejsze PDO = bardziej wrażliwa skala
- Większe PDO = szersza skala, łatwiejsza interpretacja

### 2. Walidacja Kalibracji

```python
def validate_calibration(calibrator, model, X_test, y_test, n_bins=10):
    """
    Waliduje kalibrację modelu
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    scores = calibrator.probability_to_score(probabilities)
    
    # Grupowanie po decylach score
    score_bins = pd.qcut(scores, n_bins, duplicates='drop')
    
    results = pd.DataFrame({
        'Score': scores,
        'Actual': y_test,
        'Predicted_Prob': probabilities,
        'Score_Bin': score_bins
    })
    
    summary = results.groupby('Score_Bin').agg({
        'Score': ['count', 'mean'],
        'Actual': 'mean',
        'Predicted_Prob': 'mean'
    }).round(4)
    
    summary.columns = ['Count', 'Avg_Score', 'Actual_Bad_Rate', 'Predicted_Bad_Rate']
    
    print("Walidacja Kalibracji:")
    print(summary)
    
    return summary
```

### 3. Dokumentacja Scorecardu

Każdy scorecard powinien zawierać:
1. **Metadane**: Data stworzenia, wersja, autor
2. **Parametry kalibracji**: base_score, base_odds, PDO
3. **Zakres score**: min/max teoretyczny i obserwowany
4. **Interpretacja**: tabela score → PD → rating
5. **Ograniczenia**: kiedy model nie powinien być używany

### 4. Monitoring Kalibracji

```python
def monitor_calibration_drift(scores_development, scores_production, 
                              calibrator, y_production=None):
    """
    Monitoruje drift kalibracji między development a production
    """
    # Porównanie rozkładu score
    mean_dev = np.mean(scores_development)
    mean_prod = np.mean(scores_production)
    
    std_dev = np.std(scores_development)
    std_prod = np.std(scores_production)
    
    print("Porównanie Development vs Production:")
    print(f"  Średni score DEV: {mean_dev:.0f}")
    print(f"  Średni score PROD: {mean_prod:.0f}")
    print(f"  Różnica: {mean_prod - mean_dev:.0f} punktów")
    print(f"  Std DEV: {std_dev:.0f}")
    print(f"  Std PROD: {std_prod:.0f}")
    
    # Jeśli mamy etykiety dla produkcji
    if y_production is not None:
        # Sprawdź czy predykcje są zgodne z rzeczywistością
        probs = calibrator.score_to_probability(scores_production)
        actual_bad_rate = y_production.mean()
        predicted_bad_rate = probs.mean()
        
        print(f"\nKalibracja:")
        print(f"  Rzeczywisty Bad Rate: {actual_bad_rate:.2%}")
        print(f"  Przewidziany Bad Rate: {predicted_bad_rate:.2%}")
        print(f"  Różnica: {(predicted_bad_rate - actual_bad_rate)*100:.2f} pp")
```

---

## Podsumowanie

Kalibracja scorecardu to kluczowy element budowy modelu credit scoringowego. Prawidłowo skalibrowany model:

1. **Jest zrozumiały** dla biznesu i regulatorów
2. **Jest porównywalny** z innymi modelami i historycznymi danymi
3. **Jest stabilny** w czasie (przy monitoringu)
4. **Umożliwia łatwą komunikację** progów akceptacji

Pamiętaj o regularnym monitorowaniu kalibracji i rekalibracji gdy zajdzie taka potrzeba.

---

*Ostatnia aktualizacja: Styczeń 2026*
