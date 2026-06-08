# License Plate AI: System Wizyjny do Rozpoznawania tablic rejestracyjnych
**Temat:** Lokalizacja i ekstrakcja tekstu z tablic rejestracyjnych pojazdów w czasie rzeczywistym.

**Autor:** Przemysław Rządkowski

**Dataset:** [Large-License-Plate-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)

---

## 1. Opis projektu i filozofia architektury
Projekt opiera się na dwuetapowym, sekwencyjnym potoku przetwarzania (pipeline), stworzonym w celu automatyzacji detekcji i odczytu tablic rejestracyjnych ze zdjęć statycznych. Główną filozofią tej architektury jest oddzielenie lokalizacji przestrzennej od samego rozpoznawania znaków. Gwarantuje to wysoką precyzję dzięki odizolowaniu obszaru zainteresowania przed uruchomieniem algorytmów OCR.

### 1.1. Detekcja i lokalizacja: PlateLocNet jako przestrzenny filtr sygnału
Zamiast przekazywać cały surowy obraz bezpośrednio do silnika rozpoznawania tekstu, co generowałoby ogromne narzuty obliczeniowe oraz szum. System wykorzystuje dedykowaną, autorską architekturę głębokiego uczenia o nazwie `PlateLocNet`.

* **Filtrowanie obszaru zainteresowania (ROI):** Model przetwarza obraz wejściowy, mapując go na siatkę o wymiarach 14 x 14, w celu oszacowania pewności precyzyjnego wyznaczenia współrzędnych ramki otaczającej (bounding box), bazując na podejściu znanym z architektury YOLO.
* **Izolacja tła:** Po obliczeniu dokładnych współrzędnych ramki, potok wycina dany obszar z obrazu i nakłada ciasny, dynamiczny margines (3% na osi X, 5% na osi Y). Dzięki temu kolejne etapy przetwarzania tekstu otrzymują "czystą" strukturę samej tablicy, bez zakłóceń w postaci atrap chłodnicy, zderzaków czy elementów otoczenia.

### 1.2. Ekstrakcja znaków: Integracja z silnikiem EasyOCR
Gdy obszar tablicy zostanie poprawnie wycięty i odizolowany od reszty kadru, jest traktowany jako niezależny obiekt tekstowy o wysokiej gęstości danych.

* **Ukierunkowany OCR:** Wycięty fragment obrazu trafia bezpośrednio do silnika `EasyOCR`, skonfigurowanego pod kątem odczytu znaków alfanumerycznych.
* **Ocena pewności (Confidence Scoring):** Potok analizuje potencjalne warianty odczytu, wybiera segment tekstu o najwyższym współczynniku pewności, oczyszcza go z ewentualnych spacji i formatuje znaki do znormalizowanego ciągu wielkich liter.

---

## 2. Dokumentacja techniczna

### Architektura systemu
* **Model lokalizacji:** Autorska sieć `PlateLocNet` wykonująca regresję lokalną w celu przewidywania współrzędnych w oparciu o strukturalną siatkę komórek.
* **Silnik OCR:** Biblioteka `EasyOCR` realizująca procesy przetwarzania tekstu na wyciętych obszarach.
* **Transformacje wejściowe:** Obrazy wejściowe są normalizowane i skalowane przy użyciu biblioteki `Albumentations` zgodnie ze standardami zestawów treningowych:
  * Wymiary (Shape): `(224, 224, 3)`
  * Średnia (Mean): `(0.485, 0.456, 0.406)`
  * Odchylenie std (Std): `(0.229, 0.224, 0.225)`

---

## 3. REST API & Interfejs webowy
Projekt wykorzystuje framework **FastAPI**, który pełni funkcję wydajnego, asynchronicznego mostu pomiędzy środowiskiem uruchomieniowym PyTorch a warstwą frontendową:

* **Endpoint:** `POST /analyze`
* **Body:** `image` (UploadFile)

Łańcuch przetwarzania:
1. Odbiera binarne pakiety danych obrazu i dekoduje je do macierzy BGR systemu OpenCV.
2. Uruchamia sekwencję wykonawczą (Detekcja `PlateLocNet` -> Logika wycinania -> Odczyt `EasyOCR`).
3. Konwertuje wymiary ramki otaczającej na ułamkowe pozycje CSS (`left`, `top`, `width`, `height`), co pozwala na natywne, absolutne pozycjonowanie ramek w widoku UI.
4. Zwraca odczytany tekst, pewność identyfikacji oraz współrzędne ramki w ujednoliconym schemacie JSON.

---

## 4. Jak skonfigurować projekt
### Pobieranie wag modelu:
Wagi modelu są przechowywane zewnętrznie.
1. Pobierz pliki `.pth` z folderu: [Google Drive - Plate Detection Models](https://drive.google.com/drive/folders/1IxsYJaEtT-blY_YhUpbg2-NTofaMOAAp?usp=sharing)
2. Umieść pobrane pliki w katalogu: `src/models/`

### Instalacja zależności:
Zalecane jest użycie środowiska wirtualnego (Python 3.10+):
```bash
python -m venv .venv
```
### Windows:
```bash
.venv\Scripts\activate
```
### Linux/Mac:
```bash
source .venv/bin/activate
```
### Requirements:
```bash
pip install -r requirements.txt
```

---

## 5. Jak uruchomić aplikacje
Przejdź do katalogu z API i uruchom serwer uvicorn:
```bash
uvicorn app.app:app --reload
```
Po wykonaniu tych kroków aplikacja będzie dostępna w przeglądarce pod adresem: http://localhost:8000/
