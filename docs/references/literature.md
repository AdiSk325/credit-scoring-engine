# Literatura i Materiały Referencyjne dla Credit Scoringu

## Spis Treści

1. [Klasyka Gatunku (Must Read)](#klasyka-gatunku-must-read)
2. [Dokumenty Regulacyjne](#dokumenty-regulacyjne)
3. [Papers Akademickie](#papers-akademickie)
4. [Podręczniki Uzupełniające](#podręczniki-uzupełniające)
5. [Zasoby Online](#zasoby-online)

---

## Klasyka Gatunku (Must Read)

### 1. Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards"

**ISBN**: 978-1119279150  
**Wydawca**: Wiley

#### Znaczenie
Absolutna podstawa credit scoringu. Autor, Naeem Siddiqi, jest jednym z najbardziej uznanych praktyków w branży. Książka łączy teorię z praktycznym doświadczeniem z wdrożeń w dużych instytucjach finansowych.

#### Kluczowe Rozdziały
| Rozdział | Temat | Znaczenie |
|----------|-------|-----------|
| 2 | Data Preparation | Przygotowanie danych, obsługa braków |
| 3 | Binning & WoE | Szczegółowy opis Weight of Evidence |
| 4 | Feature Selection | Metody selekcji zmiennych |
| 5 | Model Development | Regresja logistyczna dla scoringu |
| 6 | Scaling | Points to Double Odds, formuła skalowania |
| 7 | Validation | Walidacja modelu |
| 8 | Implementation | Wdrożenie i monitoring |

#### Cytowane Formuły
- **WoE**: `WoE = ln(Distribution of Goods / Distribution of Bads)`
- **IV**: `IV = Σ (Dist Good - Dist Bad) × WoE`
- **Scaling**: `Score = Offset + Factor × ln(odds)`

---

### 2. Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). "Credit Scoring and Its Applications"

**ISBN**: 978-0898714838  
**Wydawca**: SIAM (Society for Industrial and Applied Mathematics)

#### Znaczenie
Bardziej akademickie podejście. Autorzy są profesorami uniwersytetów z bogatym doświadczeniem badawczym. Książka dostarcza solidnych matematycznych podstaw.

#### Kluczowe Rozdziały
| Rozdział | Temat | Znaczenie |
|----------|-------|-----------|
| 3 | Statistical Methods | Regresja logistyczna, LDA |
| 4 | Machine Learning | Drzewa, sieci neuronowe |
| 5 | Reject Inference | Metody wnioskowania o odrzuconych |
| 6 | Economics | Ekonomika credit scoringu |
| 7 | Profit Scoring | Scoring oparty na zysku |

#### Cytat z Książki
> "The reject inference problem is one of the most vexing in credit scoring. Ignoring it can lead to seriously biased models, but the various methods proposed to solve it have their own theoretical and practical difficulties."

---

### 3. Hand, D. J., & Henley, W. E. (1997). "Statistical Classification Methods in Consumer Credit Scoring: A Review"

**Journal**: Journal of the Royal Statistical Society, Series A, 160(3), 523-541  
**DOI**: 10.1111/j.1467-985X.1997.00078.x

#### Znaczenie
Seminal paper porównujący różne metody statystyczne w kontekście credit scoringu. Często cytowany (ponad 2000 cytowań).

#### Kluczowe Wnioski
1. **Regresja logistyczna** pozostaje metodą referencyjną
2. **Sieci neuronowe** nie dają znaczącej przewagi, przy utracie interpretowalności
3. **Drzewa decyzyjne** użyteczne dla segmentacji
4. **Metodologia jest ważniejsza niż algorytm** - dobra inżynieria cech pokonuje zaawansowane modele

#### Tabela Porównawcza z Papieru
| Metoda | Interpretowalność | Dokładność | Łatwość implementacji |
|--------|-------------------|------------|----------------------|
| Regresja logistyczna | Wysoka | Dobra | Łatwa |
| Analiza dyskryminacyjna | Średnia | Dobra | Średnia |
| Drzewa decyzyjne | Bardzo wysoka | Średnia | Łatwa |
| Sieci neuronowe | Niska | Dobra | Trudna |

---

## Dokumenty Regulacyjne

### Basel Committee on Banking Supervision (BCBS)

#### Basel II: International Convergence of Capital Measurement and Capital Standards (2004)
- **Część**: Pillar 1 - Minimum Capital Requirements
- **Sekcja**: Internal Ratings-Based Approach
- **Znaczenie**: Definiuje wymogi dla modeli PD używanych do kalkulacji kapitału

**Kluczowe Definicje**:
- **PD (Probability of Default)**: Prawdopodobieństwo defaultu w horyzoncie 1 roku
- **LGD (Loss Given Default)**: Strata przy defaultcie
- **EAD (Exposure at Default)**: Ekspozycja przy defaultcie
- **Default**: Opóźnienie płatności > 90 dni lub stwierdzenie niewypłacalności

#### Basel III: A Global Regulatory Framework (2010-2011)
- **Część**: Capital Requirements
- **Znaczenie**: Zaostrzenie wymogów kapitałowych po kryzysie 2008

---

### European Banking Authority (EBA)

#### Guidelines on PD Estimation, LGD Estimation and Treatment of Defaulted Exposures (EBA/GL/2017/16)

**Data**: Listopad 2017  
**Status**: Obowiązujące

**Kluczowe Wymogi**:
1. **Długość historii**: Minimum 5 lat danych (idealne 7 lat obejmujące cały cykl ekonomiczny)
2. **Definicja Default**: Harmonizacja z Basel - 90 dni opóźnienia
3. **Margin of Conservatism**: Bufor bezpieczeństwa dla niepewności szacunków
4. **Walidacja**: Wymóg regularnej walidacji modeli

#### Final Report on Guidelines on Credit Risk Mitigation (EBA/GL/2020/05)

**Data**: Maj 2020

**Znaczenie**: Wytyczne dotyczące zabezpieczeń i ich wpływu na LGD

---

### Financial Conduct Authority (FCA) - UK

#### Fair Treatment of Customers

**Znaczenie**: Etyczne aspekty credit scoringu
- Zakaz dyskryminacji
- Przejrzystość decyzji kredytowych
- Prawo do wyjaśnienia decyzji

---

## Papers Akademickie

### Reject Inference

1. **Banasik, J., Crook, J., & Thomas, L. (2003). "Sample selection bias in credit scoring models"**
   - Journal of the Operational Research Society, 54(8), 822-832
   - Analiza wpływu sample bias na modele scoringowe
   - Porównanie metod reject inference

2. **Crook, J., & Banasik, J. (2004). "Does reject inference really improve the performance of application scoring models?"**
   - Journal of Banking & Finance, 28(4), 857-874
   - Krytyczna analiza metod reject inference
   - Wniosek: korzyści są ograniczone

3. **Feelders, A. (2000). "Credit scoring and reject inference with mixture models"**
   - International Journal of Intelligent Systems in Accounting, Finance & Management, 9(1), 1-8
   - Podejście EM (Expectation-Maximization)

### Weight of Evidence i Binning

4. **Anderson, R. (2007). "The Credit Scoring Toolkit"**
   - Oxford University Press
   - Szczegółowy opis metodologii WoE
   - Praktyczne aspekty implementacji

5. **Mironchyk, P., & Tchistiakov, V. (2017). "Monotone Optimal Binning Algorithm for Credit Risk Modeling"**
   - Theoretical foundations for optimal binning
   - Mixed Integer Programming approach

### Machine Learning w Credit Scoringu

6. **Lessmann, S., et al. (2015). "Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research"**
   - European Journal of Operational Research, 247(1), 124-136
   - Porównanie 41 klasyfikatorów
   - Wniosek: Random Forest i Gradient Boosting konkurencyjne z regresją logistyczną

7. **Kvamme, H., et al. (2018). "Predicting mortgage default using convolutional neural networks"**
   - Expert Systems with Applications, 102, 207-217
   - Nowoczesne podejście CNN

### Stabilność i Monitoring

8. **Siddiqi, N. (2012). "Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring"**
   - Rozdział o PSI i CSI
   - Metodologia monitoringu

---

## Podręczniki Uzupełniające

### Statystyka i Machine Learning

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"**
   - Bezpłatny PDF: https://hastie.su.domains/ElemStatLearn/
   - Rozdział 4: Linear Methods for Classification

2. **James, G., et al. (2021). "An Introduction to Statistical Learning"**
   - Bezpłatny PDF: https://www.statlearning.com/
   - Łagodniejsze wprowadzenie niż ESL

### Python i Implementacja

3. **McKinney, W. (2022). "Python for Data Analysis, 3rd Edition"**
   - O'Reilly
   - Pandas, podstawy analizy danych

4. **Müller, A. C., & Guido, S. (2016). "Introduction to Machine Learning with Python"**
   - O'Reilly
   - Praktyczne wprowadzenie do sklearn

---

## Zasoby Online

### Repozytoria GitHub

| Nazwa | Link | Opis |
|-------|------|------|
| optbinning | [guillermo-navas-palencia/optbinning](https://github.com/guillermo-navas-palencia/optbinning) | State-of-the-art binning |
| scorecardpy | [shichenxie/scorecardpy](https://github.com/shichenxie/scorecardpy) | Port z R |
| Toad | [amphibian-dev/toad](https://github.com/amphibian-dev/toad) | Chiński fintech |

### Kursy i Tutoriale

1. **Coursera: Credit Risk Modeling in Python**
   - DataCamp
   - Praktyczne wprowadzenie

2. **Risk Management Institute - NUS**
   - Zaawansowane kursy risk management

### Blogi i Artykuły

1. **Towards Data Science**
   - Wiele artykułów o credit scoringu
   - Search: "credit scoring python"

2. **Risk.net**
   - Profesjonalne artykuły branżowe

---

## Jak Korzystać z Tej Listy

### Dla Początkujących
1. Zacznij od **Siddiqi (2017)** - rozdziały 1-4
2. Przeczytaj **Hand & Henley (1997)** dla kontekstu
3. Zapoznaj się z dokumentacją **optbinning**

### Dla Zaawansowanych
1. Przestudiuj **Thomas et al. (2002)** - szczególnie reject inference
2. Przeczytaj papers o machine learning w scoringu
3. Zapoznaj się z dokumentami regulacyjnymi

### Dla Implementatorów
1. Przeanalizuj kod **optbinning**, **scorecardpy**, **Toad**
2. Zaimplementuj podstawowy pipeline
3. Dodawaj zaawansowane funkcjonalności iteracyjnie

---

*Ostatnia aktualizacja: Styczeń 2026*
