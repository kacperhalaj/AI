# Temat 4 - Plansza o wymiarach 4x4, wygrywa osoba, która pierwsza ułoży cztery X

# Cel i zakres projektu
Celem projektu jest implementacja gry Kółko i Krzyżyk na planszy 4x4, w której jeden z graczy jest sterowany przez sztuczną inteligencję (AI). AI wykorzystuje algorytm MiniMax z cięciami alfa-beta do podejmowania decyzji o ruchach. Projekt obejmuje logikę gry, implementację algorytmu AI oraz prosty interfejs tekstowy do interakcji z użytkownikiem.

# Opis gry
Jest to modyfikacja klasycznej gry w Kółko i Krzyżyk, rozgrywana na planszy o wymiarach 4x4. Dwóch graczy ("O" - człowiek, "X" - AI) na przemian stawia swoje symbole w pustych polach planszy. Celem gry jest ułożenie czterech swoich symboli w jednej linii - poziomo, pionowo lub na ukos (w tym na przekątnych głównych). Gracz, który pierwszy osiągnie ten cel, wygrywa. Jeśli wszystkie pola na planszy zostaną zapełnione, a żaden gracz nie ułożył czterech symboli w linii, gra kończy się remisem.

# Zastosowane rozwiązania
*   **Reprezentacja planszy:** Plansza gry jest reprezentowana jako lista list (macierz 4x4), gdzie `None` oznacza puste pole, "O" oznacza symbol gracza, a "X" symbol AI.
*   **Algorytm MiniMax:** Sercem AI jest algorytm MiniMax, który przeszukuje drzewo możliwych stanów gry, aby znaleźć optymalny ruch.
    *   **Funkcja oceniająca (`evaluate`):** Prosta funkcja heurystyczna oceniająca końcowe stany gry. Zwraca `10 - depth` dla wygranej gracza "O", `depth - 10` dla wygranej gracza "X" (AI) i `0` dla remisu lub stanu niekońcowego. Uwzględnienie głębokości (`depth`) motywuje AI do wygrywania jak najszybciej i odwlekania przegranej.
    *   **Cięcia Alfa-Beta:** Zastosowano optymalizację alfa-beta, która znacząco redukuje liczbę przeszukiwanych węzłów w drzewie gry, odcinając gałęzie, które na pewno nie wpłyną na ostateczny wybór ruchu.
    *   **Ograniczenie głębokości:** Algorytm przeszukuje drzewo do określonej głębokości (`max_depth`). Głębokość ta jest dynamicznie zwiększana w późniejszej fazie gry (`ai_turn`), aby umożliwić głębszą analizę, gdy liczba możliwych ruchów maleje. Początkowa maksymalna głębokość to `3`.
    *   **Generowanie ruchów (`get_possible_moves`):** Funkcja zwraca listę wszystkich możliwych do wykonania ruchów (pustych pól na planszy).
*   **Interfejs użytkownika:** Zaimplementowano prosty interfejs tekstowy, który wyświetla stan planszy po każdym ruchu i pobiera ruchy od gracza za pomocą konsoli.

# Przykładowa implementacja gry
Poniżej przedstawiona jest przykładowa implementacja kodu klasy TicTacToe służącej do gry w Kółko i Krzyżyk na planszy 4x4. Program umożliwia rozgrywkę pomiędzy graczem ("O") a sztuczną inteligencją ("X") wykorzystującą algorytm Minimax z optymalizacją przez przycinanie alfa-beta.

Główne funkcjonalności:
   *   Dynamiczne tworzenie planszy 4x4.
   *   Wykrywanie końca gry (wygrana, remis).
   *   Dynamiczne wyszukiwanie zwycięskich kombinacji (wiersze, kolumny, przekątne).
   *   AI analizujące możliwe ruchy przy pomocy algorytmu Minimax.
   *   Obsługa ruchu gracza oraz ruchu AI.
   *   Gra prowadzona w konsoli, z wypisywaniem aktualnego stanu planszy.

**Przykładowa symulacja programu**
Poniżej przedstawione zostały etapy gry do momentu wykrycia końca gry:
![image](https://github.com/user-attachments/assets/8ded3132-08a4-48ed-9c6b-e421121b860a)

![image](https://github.com/user-attachments/assets/92c6945d-20db-4421-bbd8-4c989e7e54ec)

![image](https://github.com/user-attachments/assets/18840fe1-588e-4cd8-b740-0b04e6641320)

![image](https://github.com/user-attachments/assets/7271feed-2574-4919-9327-0baab6e97a34)

Na przedstawionych grafikach gra zakończyła się remisem. Jak możemy zauważyć poruszanie się po planszy jest proste i intuicyjne. Dezycje komputera są szybskie i próbuja prowadzić do odniesienia przez gracza porażki.
Na początkowych fazach gry czas oczekiwania na ruch AI były znacznie wyższe niż przy ich końcu. Pierwsza dezycja została podjęta po 0.0056 sekund, 
co oczywiście jest szybkim czasem reakcji natomist dzięki zastosowaniu algorytmu MinMax wraz z cięciami alfa - beta, ostatnie 3 tury zajmowały komputerowi 0.000 sekund. 
Sztuczna inteligencja podejmuje szybkie i przemyślane decyzje, które mają utrudnić gre graczowi jednocześnie przybliżyć do wygranej AI, minimalizując z każdą turą czas oczekiwania na ruch.

# Wnioski
Implementacja skutecznie wykorzystuje algorytm MiniMax z cięciami alfa-beta do stworzenia kompetentnego przeciwnika w grze Kółko i Krzyżyk 4x4. Zastosowanie cięć alfa-beta jest kluczowe dla zapewnienia rozsądnego czasu odpowiedzi AI, zwłaszcza na większej planszy 4x4. Dynamiczne dostosowywanie głębokości przeszukiwania pozwala na balans między szybkością a jakością decyzji AI w różnych fazach gry.

**Możliwe udoskonalenia:**
*   Implementacja bardziej zaawansowanej funkcji oceniającej, która uwzględniałaby nie tylko stany końcowe, ale także pośrednie (np. liczbę linii z dwoma/trzema symbolami gracza/AI).
*   Zastosowanie "księgi otwarć" (opening book) dla przyspieszenia początkowych ruchów AI.
*   Optymalizacja kolejności przeszukiwania ruchów (np. sprawdzanie ruchów prowadzących do wygranej lub blokujących przeciwnika w pierwszej kolejności).
*   Stworzenie interfejsu graficznego (GUI) dla bardziej przyjaznej rozgrywki.
