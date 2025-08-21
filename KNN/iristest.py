import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


# Funkcja obliczająca odległość euklidesową
def euclidean_distance(x1, x2):
    return math.sqrt(sum((x1 - x2) * (x1 - x2)))


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
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray):
        # Obliczanie dokładności modelu
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def get_params(self, deep=True):
        # Zwracanie parametrów modelu (np. max_neighbors i distance_threshold)
        return {"max_neighbors": self.max_neighbors, "distance_threshold": self.distance_threshold}


# Wczytywanie danych Iris
dataset = load_iris()
X = dataset.data
y = dataset.target

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

# Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Wizualizacja danych w przestrzeni 2D za pomocą PCA
def plot_PCA(X, y, label_names, title="Wizualizacja PCA"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['target'] = y
    # Zmieniamy etykiety na ich nazwy
    df['target_name'] = [label_names[i] for i in y]
    
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='target_name', style='target_name', 
                    markers=["o", "s", "D"], palette="Set1")
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Target')
    plt.show()

plot_PCA(X_train, y_train, dataset.target_names, "Wizualizacja PCA - Zbiór Iris")

# Trening oryginalnego modelu KNeighborsClassifier
knn_original = KNeighborsClassifier(n_neighbors=5)
knn_original.fit(X_train_scaled, y_train)

# Predykcja na danych testowych przy użyciu oryginalnego modelu KNeighborsClassifier
y_pred_original = knn_original.predict(X_test_scaled)

# Obliczanie dokładności modelu oryginalnego
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"Dokładność modelu oryginalnego KNN: {accuracy_original:.2f}")

# Obliczanie macierzy pomyłek dla oryginalnego modelu
cm_original = confusion_matrix(y_test, y_pred_original)

# Wizualizacja macierzy pomyłek dla oryginalnego modelu
disp_original = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=dataset.target_names)
disp_original.plot(cmap="viridis")
plt.title("Macierz pomyłek - Oryginalny model KNN")
plt.show()

# Obliczanie precyzji, recall i F1-score dla oryginalnego modelu
precision_original = precision_score(y_test, y_pred_original, average='weighted')
recall_original = recall_score(y_test, y_pred_original, average='weighted')
f1_original = f1_score(y_test, y_pred_original, average='weighted')

print(f"Precyzja modelu oryginalnego: {precision_original:.2f}")
print(f"Recall modelu oryginalnego: {recall_original:.2f}")
print(f"F1-score modelu oryginalnego: {f1_original:.2f}")

# Trening modelu KNN z dynamiczną liczbą sąsiadów
model = KNN(max_neighbors=10, distance_threshold=0.5)
model.fit(X_train_scaled, y_train)

# Predykcja na danych testowych przy użyciu nowego modelu KNN
y_pred = model.predict(X_test_scaled)

# Obliczanie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu (Nowa implementacja z dynamiczną liczbą sąsiadów): {accuracy:.2f}")

# Obliczanie macierzy pomyłek
cm = confusion_matrix(y_test, y_pred)

# Wizualizacja macierzy pomyłek dla nowego modelu
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
disp.plot(cmap="viridis")
plt.title("Macierz pomyłek - Nowy model z dynamiczną liczbą sąsiadów KNN")
plt.show()

# Obliczanie precyzji, recall i F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precyzja: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Walidacja krzyżowa
cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f'Wyniki walidacji krzyżowej: {cross_val_scores}')

# Porównanie wyników: Nowy model vs. oryginalny KNN
print("\nPorównanie wyników:")
print(f"Dokładność - Oryginalny model: {accuracy_original:.2f}, Nowy model: {accuracy:.2f}")
print(f"Precyzja - Oryginalny model: {precision_original:.2f}, Nowy model: {precision:.2f}")
print(f"Recall - Oryginalny model: {recall_original:.2f}, Nowy model: {recall:.2f}")
print(f"F1-score - Oryginalny model: {f1_original:.2f}, Nowy model: {f1:.2f}")

# Podsumowanie wyników
print("\nPodsumowanie wyników:")
if accuracy > accuracy_original:
    print("Nowy model ma wyższą dokładność niż oryginalny KNN.")
elif accuracy < accuracy_original:
    print("Oryginalny model KNN ma wyższą dokładność niż Nowy model.")
else:
    print("Dokładność obu modeli jest taka sama.")

if f1 > f1_original:
    print("Nowy model ma wyższy F1-score niż oryginalny KNN.")
elif f1 < f1_original:
    print("Oryginalny model KNN ma wyższy F1-score niż Nowy model.")
else:
    print("F1-score obu modeli jest taki sam.")






# Porównanie wyników
metrics = ['Dokładność', 'Precyzja', 'Recall', 'F1-score']
original_model_scores = [accuracy_original, precision_original, recall_original, f1_original]
new_model_scores = [accuracy, precision, recall, f1]

# Tworzenie wykresu słupkowego porównującego wyniki
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35  # Szerokość słupków

bars1 = ax.bar(x - width/2, original_model_scores, width, label='Oryginalny KNN', color='b')
bars2 = ax.bar(x + width/2, new_model_scores, width, label='Nowy model KNN', color='g')

ax.set_xlabel('Metryki')
ax.set_ylabel('Wartość')
ax.set_title('Porównanie wyników: Oryginalny KNN vs Nowy model KNN')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Dodanie wartości na słupkach
def add_values(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

add_values(bars1)
add_values(bars2)

plt.tight_layout()
plt.show()

# Wizualizacja macierzy pomyłek
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Wizualizacja macierzy pomyłek dla oryginalnego modelu
disp_original = ConfusionMatrixDisplay(confusion_matrix=cm_original, display_labels=dataset.target_names)
disp_original.plot(ax=ax[0], cmap="Blues")
ax[0].set_title("Macierz pomyłek - Oryginalny KNN")

# Wizualizacja macierzy pomyłek dla nowego modelu
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
disp.plot(ax=ax[1], cmap="Blues")
ax[1].set_title("Macierz pomyłek - Nowy model KNN")

plt.tight_layout() 
plt.show() 
