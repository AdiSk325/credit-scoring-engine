# Reject Inference - Wnioskowanie o Odrzuconych

## Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Problem Sample Bias](#problem-sample-bias)
3. [Metody Reject Inference](#metody-reject-inference)
4. [Implementacja w Pythonie](#implementacja-w-pythonie)
5. [Porównanie Metod](#porównanie-metod)
6. [Best Practices](#best-practices)
7. [Literatura](#literatura)

---

## Wprowadzenie

Reject Inference to zbiór technik służących do włączenia informacji o odrzuconych wnioskach kredytowych do procesu budowy modelu scoringowego. Jest to jeden z najważniejszych i najbardziej kontrowersyjnych tematów w credit scoringu.

### Dlaczego Jest To Ważne?

Budując model scoringowy na podstawie historycznych danych, mamy dostęp tylko do informacji o klientach, którym udzielono kredytu (bo tylko dla nich znamy wynik - spłacili czy nie). Klienci odrzuceni są niewidoczni dla modelu, co może prowadzić do **błędu selekcji (sample bias)**.

---

## Problem Sample Bias

### Wizualizacja Problemu

```
Populacja Wnioskujących o Kredyt
│
├── Zaakceptowani (60%)
│   ├── Good - spłacili (55%)    ← Mamy etykietę
│   └── Bad - nie spłacili (5%)  ← Mamy etykietę
│
└── Odrzuceni (40%)
    ├── Good - spłaciliby (?)    ← Brak etykiety
    └── Bad - nie spłaciliby (?) ← Brak etykiety
```

### Konsekwencje Ignorowania Odrzuconych

1. **Przesunięcie rozkładu zmiennych** - Rozkład zmiennych w próbie akceptowanych różni się od rozkładu w całej populacji
2. **Zaniżone estymaty współczynników** - Model może niedoszacowywać wpływu niektórych zmiennych
3. **Nadmierna pewność modelu** - Model może być zbyt pewny swoich predykcji
4. **Selection Bias w czasie** - Jeśli polityka akceptacji się zmieni, model może przestać działać

### Kiedy Problem Jest Szczególnie Poważny?

| Scenariusz | Nasilenie Problemu |
|------------|-------------------|
| Niska stopa akceptacji (<50%) | Bardzo wysokie |
| Restrykcyjna polityka kredytowa | Wysokie |
| Zmiana polityki akceptacji w czasie | Wysokie |
| Wysoka stopa akceptacji (>80%) | Niskie |
| Stabilna polityka akceptacji | Średnie |

---

## Metody Reject Inference

### 1. Ignorowanie (No Reject Inference)

**Opis**: Budowa modelu wyłącznie na zaakceptowanych.

**Kiedy stosować**:
- Wysoka stopa akceptacji (>80%)
- Brak zmiany polityki akceptacji
- Pilotażowe modele

**Wady**:
- Sample bias

**Kod**:
```python
# Po prostu używamy tylko danych akceptowanych
model.fit(X_accepted, y_accepted)
```

---

### 2. Hard Cut-off

**Opis**: Przypisanie etykiety "Bad" wszystkim odrzuconym.

**Założenie**: Odrzuceni zostali odrzuceni słusznie - wszyscy są złymi klientami.

**Kiedy stosować**:
- Bardzo restrykcyjna polityka akceptacji oparta na wyraźnych sygnałach ryzyka
- Chcemy konserwatywnego modelu (lepiej odrzucić dobrego niż zaakceptować złego)

**Wady**:
- Zbyt pesymistyczne
- Może prowadzić do overfitting na zmiennych skorelowanych z decyzją o odrzuceniu

**Kod**:
```python
def hard_cutoff_inference(df_accepted, df_rejected, target_col='target'):
    """
    Hard Cut-off Reject Inference
    
    Parameters:
    -----------
    df_accepted : pd.DataFrame
        Dane zaakceptowanych wniosków z etykietą Good/Bad
    df_rejected : pd.DataFrame
        Dane odrzuconych wniosków (bez etykiety)
    target_col : str
        Nazwa kolumny z etykietą
    
    Returns:
    --------
    df_combined : pd.DataFrame
        Połączone dane z przypisanymi etykietami
    """
    df_rejected = df_rejected.copy()
    df_rejected[target_col] = 1  # Wszystkich odrzuconych uznajemy za Bad
    
    df_combined = pd.concat([df_accepted, df_rejected], ignore_index=True)
    
    return df_combined
```

---

### 3. Simple Augmentation (Reweighting)

**Opis**: Przypisanie odrzuconym etykiety zgodnej z predykcją istniejącego modelu, bez modyfikacji wag.

**Założenie**: Istniejący model jest wystarczająco dobry do klasyfikacji odrzuconych.

**Kiedy stosować**:
- Mamy wcześniejszy model scoringowy
- Nie chcemy wprowadzać złożoności wag

**Wady**:
- Wzmacnia bias istniejącego modelu
- Wprowadza deterministyczne etykiety

**Kod**:
```python
def simple_augmentation(df_accepted, df_rejected, existing_model, 
                        features, target_col='target', threshold=0.5):
    """
    Simple Augmentation - przypisanie klasy na podstawie predykcji modelu
    
    Parameters:
    -----------
    threshold : float
        Próg prawdopodobieństwa dla klasyfikacji jako Bad
    """
    df_rejected = df_rejected.copy()
    
    # Predykcja prawdopodobieństwa Bad dla odrzuconych
    prob_bad = existing_model.predict_proba(df_rejected[features])[:, 1]
    
    # Binarna klasyfikacja
    df_rejected[target_col] = (prob_bad >= threshold).astype(int)
    
    df_combined = pd.concat([df_accepted, df_rejected], ignore_index=True)
    
    return df_combined
```

---

### 4. Parceling

**Opis**: Losowe przypisywanie klasy Good/Bad w oparciu o przewidywane prawdopodobieństwo.

**Założenie**: Prawdopodobieństwo Bad z modelu jest dobrym estymatem dla odrzuconych.

**Kiedy stosować**:
- Mamy wcześniejszy model
- Chcemy zachować losowość
- Akceptujemy, że wyniki mogą się różnić przy każdym uruchomieniu

**Wady**:
- Wprowadza losowość
- Wymaga ustalenia seed dla powtarzalności

**Kod**:
```python
def parceling_inference(df_accepted, df_rejected, existing_model, 
                        features, target_col='target', random_state=42):
    """
    Parceling Reject Inference
    
    Parameters:
    -----------
    random_state : int
        Seed dla powtarzalności wyników
    """
    rng = np.random.default_rng(random_state)
    df_rejected = df_rejected.copy()
    
    # Predykcja prawdopodobieństwa Bad dla odrzuconych
    prob_bad = existing_model.predict_proba(df_rejected[features])[:, 1]
    
    # Losowe przypisanie klasy zgodnie z prawdopodobieństwem
    random_vals = rng.uniform(0, 1, size=len(df_rejected))
    df_rejected[target_col] = (random_vals < prob_bad).astype(int)
    
    df_combined = pd.concat([df_accepted, df_rejected], ignore_index=True)
    
    return df_combined
```

---

### 5. Fuzzy Augmentation

**Opis**: Każdy odrzucony jest reprezentowany jako częściowo Good i częściowo Bad poprzez system wag.

**Założenie**: Niepewność klasyfikacji powinna być zachowana poprzez wagi.

**Kiedy stosować**:
- Chcemy zachować pełną informację o niepewności
- Algorytm uczący obsługuje `sample_weight`
- Ważna jest stabilność wyników

**Wady**:
- Zwiększa rozmiar zbioru danych (duplikacja)
- Wymaga algorytmów obsługujących wagi

**Kod**:
```python
def fuzzy_augmentation(df_accepted, df_rejected, existing_model, 
                       features, target_col='target', weight_col='sample_weight'):
    """
    Fuzzy Augmentation Reject Inference
    
    Każdy odrzucony klient jest duplikowany:
    - raz jako Bad z wagą = P(Bad)
    - raz jako Good z wagą = 1 - P(Bad)
    """
    df_rejected = df_rejected.copy()
    
    # Predykcja prawdopodobieństwa Bad
    prob_bad = existing_model.predict_proba(df_rejected[features])[:, 1]
    
    # Tworzenie zduplikowanych rekordów z wagami
    df_rejected_bad = df_rejected.copy()
    df_rejected_bad[target_col] = 1
    df_rejected_bad[weight_col] = prob_bad
    
    df_rejected_good = df_rejected.copy()
    df_rejected_good[target_col] = 0
    df_rejected_good[weight_col] = 1 - prob_bad
    
    # Akceptowani dostają wagę 1
    df_accepted = df_accepted.copy()
    df_accepted[weight_col] = 1.0
    
    # Łączenie wszystkich danych
    df_combined = pd.concat([df_accepted, df_rejected_bad, df_rejected_good], 
                            ignore_index=True)
    
    return df_combined
```

---

### 6. Extrapolation Method

**Opis**: Budowa modelu na akceptowanych, ekstrapolacja predykcji na odrzuconych.

**Założenie**: Model wytrenowany na akceptowanych generalizuje na odrzuconych.

**Wady**:
- Ekstrapolacja poza zakres danych treningowych jest ryzykowna
- Model może nie działać dobrze dla ekstremalnych przypadków

**Kod**:
```python
def extrapolation_method(df_accepted, df_rejected, features, target_col='target',
                         n_iterations=3):
    """
    Iteracyjna metoda ekstrapolacji
    
    1. Buduj model na akceptowanych
    2. Przypisz etykiety odrzuconym
    3. Przebuduj model na wszystkich
    4. Powtórz
    """
    from sklearn.linear_model import LogisticRegression
    
    df_rejected = df_rejected.copy()
    
    # Pierwszy model tylko na akceptowanych
    model = LogisticRegression(max_iter=1000)
    model.fit(df_accepted[features], df_accepted[target_col])
    
    for i in range(n_iterations):
        # Przypisz etykiety odrzuconym
        prob_bad = model.predict_proba(df_rejected[features])[:, 1]
        df_rejected[target_col] = (prob_bad >= 0.5).astype(int)
        
        # Połącz dane
        df_combined = pd.concat([df_accepted, df_rejected], ignore_index=True)
        
        # Przebuduj model
        model.fit(df_combined[features], df_combined[target_col])
        
        print(f"Iteracja {i+1}: Bad rate wśród odrzuconych = {df_rejected[target_col].mean():.2%}")
    
    return df_combined, model
```

---

### 7. Mixture Model (EM Algorithm)

**Opis**: Wykorzystanie algorytmu EM (Expectation-Maximization) do estymacji etykiet odrzuconych.

**Założenie**: Dane pochodzą z mieszaniny dwóch rozkładów (Good i Bad).

**Kiedy stosować**:
- Zaawansowane zastosowania
- Gdy inne metody nie działają

**Wady**:
- Złożona implementacja
- Wrażliwość na inicjalizację
- Może nie zbiegać do optymalnego rozwiązania

**Kod** (uproszczony):
```python
def mixture_model_inference(df_accepted, df_rejected, features, target_col='target',
                            n_iterations=100, tol=1e-4):
    """
    EM Algorithm dla Reject Inference (uproszczona wersja)
    """
    from sklearn.linear_model import LogisticRegression
    
    df_rejected = df_rejected.copy()
    
    # Inicjalizacja - model na akceptowanych
    model = LogisticRegression(max_iter=1000)
    model.fit(df_accepted[features], df_accepted[target_col])
    
    prev_log_likelihood = -np.inf
    
    for i in range(n_iterations):
        # E-step: estymuj prawdopodobieństwa dla odrzuconych
        prob_bad = model.predict_proba(df_rejected[features])[:, 1]
        
        # M-step: przebuduj model z wagami
        X_combined = pd.concat([df_accepted[features], df_rejected[features]])
        
        # Dla akceptowanych - znane etykiety
        y_accepted = df_accepted[target_col]
        weights_accepted = np.ones(len(df_accepted))
        
        # Dla odrzuconych - wagi oparte na prawdopodobieństwie
        # Duplikujemy odrzuconych dla obu klas
        X_rejected_bad = df_rejected[features].copy()
        y_rejected_bad = np.ones(len(df_rejected))
        weights_rejected_bad = prob_bad
        
        X_rejected_good = df_rejected[features].copy()
        y_rejected_good = np.zeros(len(df_rejected))
        weights_rejected_good = 1 - prob_bad
        
        # Łączymy wszystko
        X_full = pd.concat([df_accepted[features], X_rejected_bad, X_rejected_good])
        y_full = np.concatenate([y_accepted, y_rejected_bad, y_rejected_good])
        weights_full = np.concatenate([weights_accepted, weights_rejected_bad, weights_rejected_good])
        
        # Trenuj model
        model.fit(X_full, y_full, sample_weight=weights_full)
        
        # Sprawdź zbieżność (uproszczone)
        # W pełnej implementacji używamy log-likelihood
        
    return model
```

---

## Porównanie Metod

### Tabela Porównawcza

| Metoda | Złożoność | Pesymizm | Wymaga modelu | Zachowuje niepewność | Stabilność |
|--------|-----------|----------|---------------|---------------------|------------|
| No Inference | Bardzo niska | N/A | Nie | N/A | Wysoka |
| Hard Cut-off | Niska | Bardzo wysoki | Nie | Nie | Wysoka |
| Simple Augmentation | Niska | Średni | Tak | Nie | Wysoka |
| Parceling | Niska | Średni | Tak | Częściowo | Niska |
| Fuzzy Augmentation | Średnia | Niski | Tak | Tak | Wysoka |
| Extrapolation | Średnia | Średni | Nie | Nie | Średnia |
| Mixture Model | Wysoka | Niski | Nie | Tak | Niska |

### Rekomendacje

| Scenariusz | Rekomendowana Metoda |
|------------|---------------------|
| Szybki prototyp | No Inference lub Hard Cut-off |
| Konserwatywny model | Hard Cut-off |
| Przebudowa istniejącego modelu | Fuzzy Augmentation |
| Brak wcześniejszego modelu | Extrapolation |
| Zaawansowane zastosowania | Mixture Model |

---

## Best Practices

### 1. Zawsze porównuj wyniki

Zbuduj modele z różnymi metodami reject inference i porównaj:
- Gini/AUC na populacji akceptowanych
- Rozkład score'ów
- Stabilność współczynników

### 2. Waliduj na Out-of-Time Sample

Test ostateczny - jak model działa na nowych danych:
```python
# Walidacja Out-of-Time
X_oot = df_oot[features]
y_oot = df_oot[target_col]

# Model z reject inference
prob_with_ri = model_with_ri.predict_proba(X_oot)[:, 1]
gini_with_ri = 2 * roc_auc_score(y_oot, prob_with_ri) - 1

# Model bez reject inference
prob_no_ri = model_no_ri.predict_proba(X_oot)[:, 1]
gini_no_ri = 2 * roc_auc_score(y_oot, prob_no_ri) - 1

print(f"Gini z RI: {gini_with_ri:.4f}")
print(f"Gini bez RI: {gini_no_ri:.4f}")
```

### 3. Dokumentuj metodę

W dokumentacji modelu zawsze opisz:
- Jaka metoda reject inference została użyta
- Jaki był odsetek odrzuconych
- Jak zmienił się rozkład zmiennych po włączeniu odrzuconych

### 4. Rozważ wpływ na zmienne

Niektóre zmienne mogą być silnie skorelowane z decyzją o odrzuceniu:
```python
# Sprawdź korelację z decyzją o akceptacji
df['is_accepted'] = df['application_status'].map({'accepted': 1, 'rejected': 0})

for feature in features:
    corr = df[feature].corr(df['is_accepted'])
    if abs(corr) > 0.3:
        print(f"UWAGA: {feature} silnie skorelowane z akceptacją (r={corr:.3f})")
```

---

## Literatura

1. **Banasik, J., Crook, J., & Thomas, L. (2003)**. Sample selection bias in credit scoring models. *Journal of the Operational Research Society*, 54(8), 822-832.

2. **Crook, J., & Banasik, J. (2004)**. Does reject inference really improve the performance of application scoring models? *Journal of Banking & Finance*, 28(4), 857-874.

3. **Feelders, A. (2000)**. Credit scoring and reject inference with mixture models. *International Journal of Intelligent Systems in Accounting, Finance & Management*, 9(1), 1-8.

4. **Hand, D. J. (2001)**. Reject inference in credit operations. *Handbook of Credit Scoring*, 225-240.

5. **Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002)**. *Credit Scoring and Its Applications*. SIAM. Chapter 5.

---

*Ostatnia aktualizacja: Styczeń 2026*
