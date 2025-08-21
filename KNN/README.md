# Temat 6 - Algorytm KNN z dynamiczną liczbą sąsiadów

## Opis problemu i podejścia

Algorytm k-najbliższych sąsiadów (KNN) jest prostym i skutecznym algorytmem klasyfikacji, który działa na zasadzie głosowania wśród k najbliższych sąsiadów danego obiektu. Jednym z kluczowych parametrów tego algorytmu jest stała liczba sąsiadów k, której wybór może znacząco wpłynąć na jakość klasyfikacji. Zbyt mała wartość może prowadzić do nadmiernego dopasowania do danych treningowych, podczas gdy zbyt duża może powodować zbyt duże uogólnienie.

W tym projekcie zaproponowano modyfikację standardowego algorytmu KNN, polegającą na dynamicznym doborze liczby sąsiadów dla każdego klasyfikowanego obiektu. Liczba sąsiadów nie jest stała, ale zależy od rozrzutu odległości między kolejnymi sąsiadami. Algorytm uwzględnia kolejnych najbliższych sąsiadów, dopóki różnica odległości między sąsiadami nie przekroczy ustalonego progu. Dodatkowo, aby zapobiec uwzględnianiu zbyt wielu sąsiadów, wprowadzono parametr maksymalnej liczby sąsiadów.

Przykład implementacji:

```python
class KNN:
    def __init__(self, max_neighbors=5, distance_threshold=0.5):
        # Maksymalna liczba sąsiadów oraz próg odległości
        self.max_neighbors = max_neighbors
        self.distance_threshold = distance_threshold

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Przyjęcie danych treningowych
        self.X = X
        self.y = y

    def predict(self, X_test: np.ndarray):
        # Predykcja na podstawie dynamicznej liczby sąsiadów
        predictions = []
        for x_test in X_test:
            dist = []
            for i in range(len(self.X)):
                dist.append(euclidean_distance(self.X[i], x_test))

            sorted_indices = np.argsort(dist)
            
            # Określenie liczby sąsiadów na podstawie odległości
            dynamic_neighbors = 1
            for i in range(1, len(sorted_indices)):
                if dist[sorted_indices[i]] - dist[sorted_indices[i-1]] < self.distance_threshold:
                    dynamic_neighbors += 1
                else:
                    break
            
            dynamic_neighbors = min(dynamic_neighbors, self.max_neighbors)
            labels = self.y[sorted_indices[:dynamic_neighbors]]
            
            unique, counts = np.unique(labels, return_counts=True)
            most_common = unique[np.argmax(counts)]
            predictions.append(most_common)
        
        return np.array(predictions)
```

## Szczegółowy opis metodyki

Algorytm KNN z dynamiczną liczbą sąsiadów działa w następujący sposób:

1. **Inicjalizacja parametrów**:
   - `max_neighbors`: maksymalna dozwolona liczba sąsiadów do uwzględnienia
   - `distance_threshold`: próg różnicy odległości między kolejnymi sąsiadami

2. **Proces trenowania** (metoda `fit`):
   - Zapamiętanie danych treningowych `X` oraz odpowiadających im etykiet `y`

3. **Proces predykcji** (metoda `predict`):
   - Dla każdego obiektu testowego:
     1. Obliczenie odległości euklidesowej do wszystkich obiektów treningowych
     2. Sortowanie obiektów treningowych według rosnącej odległości
     3. Dynamiczne określenie liczby sąsiadów:
        - Zaczynamy od 1 sąsiada
        - Dodajemy kolejnych sąsiadów, dopóki różnica odległości między aktualnym a poprzednim sąsiadem jest mniejsza niż `distance_threshold`
        - Ograniczamy liczbę sąsiadów do `max_neighbors`
     4. Wybór klasy poprzez głosowanie większościowe wśród dynamicznie wybranych sąsiadów

Matematycznie, dla obiektu testowego x, liczba sąsiadów k(x) jest określona jako:

k(x) = min{i : d(x, x_i) - d(x, x_{i-1}) ≥ distance_threshold lub i = max_neighbors}

gdzie:
- d(x, x_i) to odległość euklidesowa między obiektem testowym x a i-tym najbliższym sąsiadem
- x_i to i-ty najbliższy sąsiad obiektu x w zbiorze treningowym

Odległość euklidesowa między dwoma punktami x1 i x2 w przestrzeni n-wymiarowej jest obliczana jako:

d(x1, x2) = √Σ(x1_i - x2_i)²

## Opis eksperymentów

Do oceny efektywności zmodyfikowanego algorytmu KNN z dynamiczną liczbą sąsiadów przeprowadzono szereg eksperymentów na dwóch różnych zbiorach danych: Iris i Wine. Eksperymenty zostały zaprojektowane w celu porównania wydajności standardowego algorytmu KNN z stałą liczbą sąsiadów oraz zmodyfikowanego algorytmu z dynamiczną liczbą sąsiadów.

Metodyka eksperymentalna obejmowała:

1. **Podział danych**: Zastosowano metodę train_test_split do podziału danych na zbiór treningowy i testowy w proporcji 50:50, z zachowaniem stratyfikacji klas.

2. **Preprocessing danych**: Zastosowano standaryzację danych przy użyciu StandardScaler, aby sprowadzić wszystkie cechy do tej samej skali.

3. **Wizualizacja danych**: Wykorzystano analizę głównych składowych (PCA) do redukcji wymiarowości i wizualizacji danych w przestrzeni dwuwymiarowej.

4. **Trenowanie modeli**:
   - Standardowy KNN z k=5
   - Zmodyfikowany KNN z max_neighbors=10 i distance_threshold=0.5

5. **Ewaluacja modeli**: Obliczono i porównano następujące metryki:
   - Dokładność (accuracy)
   - Precyzja (precision)
   - Czułość (recall)
   - Miara F1 (F1-score)
   - Macierz pomyłek (confusion matrix)

6. **Walidacja krzyżowa**: Zastosowano 5-krotną walidację krzyżową do oceny stabilności modelu z dynamiczną liczbą sąsiadów.

### Przedstawienie zbioru danych

#### Zbiór Iris

Iris to klasyczny zbiór danych zawierający pomiary kwiatów trzech gatunków irysów (Setosa, Versicolor i Virginica). Jest to często używany zbiór danych w uczeniu maszynowym do testowania algorytmów klasyfikacji.

- **Źródło**: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/iris)
- **Wielkość**: 150 instancji (po 50 dla każdego gatunku)
- **Atrybuty warunkowe**:
  - Długość działki kielicha (sepal length) [cm]
  - Szerokość działki kielicha (sepal width) [cm]
  - Długość płatka (petal length) [cm]
  - Szerokość płatka (petal width) [cm]
- **Atrybut decyzyjny**: Gatunek irysa (Setosa, Versicolor, Virginica)

#### Zbiór Wine

Wine to zbiór danych zawierający wyniki analizy chemicznej win pochodzących z tego samego regionu we Włoszech, ale produkowanych przez trzech różnych producentów.

- **Źródło**: UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine)
- **Wielkość**: 178 instancji (59 dla klasy 1, 71 dla klasy 2, 48 dla klasy 3)
- **Atrybuty warunkowe**: 13 cech chemicznych, w tym m.in.:
  - Zawartość alkoholu
  - Kwasowość jabłkowa
  - Zawartość popiołu
  - Alkaliczność popiołu
  - Magnez
  - Fenole ogółem
  - Flawonoidy
  - Fenole nieflawonoidy
  - Proantocyjaniny
  - Intensywność koloru
  - Odcień
  - OD280/OD315 win rozcieńczonych
  - Prolina
- **Atrybut decyzyjny**: Producent wina (1, 2, 3)

### Prezentacja wyników

Poniżej przedstawiono wyniki eksperymentów dla obu zbiorów danych.

#### Zbiór Iris

Dla zbioru Iris oba modele (standardowy KNN i zmodyfikowany KNN z dynamiczną liczbą sąsiadów) osiągnęły podobne wyniki, co sugeruje, że w przypadku tego prostego zbioru danych dynamiczna liczba sąsiadów nie przynosi znaczącej poprawy.

| Metryka    | Standardowy KNN | Dynamiczny KNN |
|------------|-----------------|----------------|
| Dokładność | ~0.95           | ~0.96          |
| Precyzja   | ~0.95           | ~0.96          |
| Recall     | ~0.95           | ~0.96          |
| F1-score   | ~0.95           | ~0.96          |

Poniżej zamieszony został widok na wykresie słubkowym przedstawiający różnice pomiędzy wynikami 
standardowego KNN i jedo modyfikacją
![image](https://github.com/user-attachments/assets/fa55562b-3660-4d09-942e-059c72372785)

Kolejny wykres ukazuje porównianie wyglądu macierzy pomyłek dla obu modeli KNN. Jak możemy zauważyć macierz do prawej stronie będąca reprezentantem zmodyfikowanego modelu KNN z niego większą dokładnością przycisuje wartości dezycji do obiektów. Uwagę może przyciągnąć, że pomimo niewiele większej dokładności, większyła się również rozbieżność pomyłek.  
![image](https://github.com/user-attachments/assets/01e86256-692f-468e-a667-4cbf4fa3f0ab)

Na ostatnim obrazku znajdziemy wykres punktowym 2D jako wizualizacja PCA, na której znajdują się naniesione punkty z odpowiednimi oznaczeniem na płaszczyźnie 2D. Wykres prezentuje jak odpowiednie decyzje wartości grupują się w większe skupiska. Dowodzi to, że ta dokładności w przypisywaniu odpowiednich decyzji jest minimalnie zachwiana i niektóre obiekty mogą mieć niedokładną wartość przypisaną
![image](https://github.com/user-attachments/assets/83f21304-0fb3-4f8c-953d-1178cd8347c1)



#### Zbiór Wine

Dla bardziej złożonego zbioru Wine, dynamiczny KNN wykazał lepsze wyniki w porównaniu do standardowego KNN.

| Metryka    | Standardowy KNN | Dynamiczny KNN |
|------------|-----------------|----------------|
| Dokładność | ~0.94           | ~0.96          |
| Precyzja   | ~0.95           | ~0.96          |
| Recall     | ~0.94           | ~0.96          |
| F1-score   | ~0.94           | ~0.95          |

Poniżej znajduje się ilustracja powyższych danych w postaci wykresu słupkowego. Różnice w wynikach nieco bardziej od siebie odbiegają co bedzie skutkować w dalszych wynikach przeprowadzanych testów.
![image](https://github.com/user-attachments/assets/c299ce5d-75ac-434c-860a-4e496566188b)

Macierze pomyłek pokazują, że dynamiczny KNN lepiej radzi sobie z klasyfikacją próbek z klasy 2 i 3, co może sugerować, że dynamiczna liczba sąsiadów jest szczególnie korzystna w przypadkach, gdy granice między klasami są bardziej złożone.
![image](https://github.com/user-attachments/assets/6ca7171d-6442-4039-9fd8-6294ddf18457)

Wizualizacja PCA na wykresie punktowym, na którym przedstawione są współrzędne obiektów z odpowiadającym im oznaczeniem nie jest tam jasna w rozumieniu i jednoznaczna jak w przypadku łatwiejszego zbioru danych. Na poniższym obrazku ciężej jest dostrzeć większe skupiska punktów, co dowodzi że wyniki testów były nieco gorsze niż przy wykorzystaniu danych z irysami.
![image](https://github.com/user-attachments/assets/7dee849d-faad-4f46-832f-b1ac50948933)

Wyniki walidacji krzyżowej dla dynamicznego KNN na zbiorze Wine wykazały stabilne zachowanie modelu, z wynikami dokładności w okolicach 0.90-0.94 dla różnych podziałów danych.

## Wnioski

Na podstawie przeprowadzonych eksperymentów można wyciągnąć następujące wnioski:

1. **Efektywność dynamicznej liczby sąsiadów**: Algorytm KNN z dynamiczną liczbą sąsiadów wykazuje lepsze wyniki na bardziej złożonych zbiorach danych, takich jak Wine, gdzie granice między klasami są mniej wyraźne. W przypadku prostego zbioru Iris, różnice między standardowym KNN a dynamicznym KNN są minimalne.

2. **Adaptacyjność**: Dynamiczna liczba sąsiadów pozwala algorytmowi adaptować się do lokalnej struktury danych, co może być szczególnie korzystne w przypadku niejednorodnych zbiorów danych, gdzie różne regiony przestrzeni cech mogą wymagać różnej liczby sąsiadów do poprawnej klasyfikacji.

3. **Wpływ parametrów**: Wybór parametrów `max_neighbors` i `distance_threshold` ma istotny wpływ na wydajność algorytmu. Zbyt mała wartość `distance_threshold` może prowadzić do uwzględniania zbyt małej liczby sąsiadów, podczas gdy zbyt duża wartość może skutkować nadmiernym uogólnieniem. Parametr `max_neighbors` stanowi ważne ograniczenie, szczególnie w regionach z dużą gęstością danych.

4. **Złożoność obliczeniowa**: Algorytm z dynamiczną liczbą sąsiadów ma podobną złożoność obliczeniową do standardowego KNN, ponieważ głównym kosztem jest obliczenie odległości do wszystkich obiektów treningowych. Dodatkowy koszt dynamicznego doboru liczby sąsiadów jest nieznaczny.

5. **Potencjalne zastosowania**: Metoda dynamicznego doboru liczby sąsiadów może być szczególnie przydatna w zadaniach klasyfikacji z nierównomiernym rozkładem klas lub w przypadku danych z regionami o różnej gęstości.

6. **Przyszłe kierunki badań**: Warto rozważyć zastosowanie różnych metryk odległości oraz automatyczne dobieranie parametrów `distance_threshold` i `max_neighbors` w oparciu o charakterystykę danych, np. poprzez optymalizację tych parametrów za pomocą walidacji krzyżowej.

Podsumowując, algorytm KNN z dynamiczną liczbą sąsiadów stanowi obiecującą modyfikację standardowego KNN, która może poprawić jakość klasyfikacji, szczególnie dla złożonych zbiorów danych. Jego adaptacyjny charakter pozwala na lepsze dopasowanie do lokalnej struktury danych, co może przełożyć się na lepsze wyniki klasyfikacji.
