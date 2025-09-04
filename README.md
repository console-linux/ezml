# EZML - Easy Machine Learning Preprocessing Toolkit

A Python library for simplifying data preprocessing and machine learning workflows. EZML provides a set of tools to handle common data preprocessing tasks and streamline the machine learning pipeline.

## Features

- **String Encoding**: Automatically map string values to integers
- **Data Type Optimization**: Convert appropriate float columns to integers
- **NaN Handling**: Intelligent handling of missing values based on data types
- **Anomaly Detection**: Remove outliers using Isolation Forest
- **Encoding/Decoding**: Consistent mapping between training and test datasets
- **Visualization**: Visual representation of missing values
- **End-to-end Pipeline**: Complete solution from data loading to submission generation

## Installation

```bash
pip install pandas numpy scikit-learn yellowbrick matplotlib seaborn
```

Simply copy the `ezml.py` file into your project directory.

## Usage

### Basic Preprocessing

```python
import pandas as pd
from ezml import preprocess_df

# Load your data
df = pd.read_csv('data.csv')

# Preprocess with a single function call
processed_df = preprocess_df(df)
```

### Advanced Usage with Mappings

```python
# Get processed data along with mapping dictionaries
processed_df, mappings = preprocess_df(df, mapping=True)

# Encode new data using existing mappings
new_data_encoded = encode(new_data, mappings)

# Decode back to original values
original_data = decode(encoded_data, mappings)
```

### Complete ML Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from ezml import solve

# Set up and run complete pipeline
model = RandomForestClassifier()
x_cols = ['feature1', 'feature2', 'feature3']
y_col = 'target'

submission = solve(
    model=model,
    x_cols=x_cols,
    y_col=y_col,
    pred_col_name='predictions',
    train_df_filepath='train.csv',
    test_df_filepath='test.csv',
    submission_path='submission.csv',
    id_test_col_name='id',
    id_submission_col_name='id'
)
```

## API Reference

### Core Functions

#### `map_unique_strings(df, columns='all', return_mappings=False)`
Maps string columns to integer representations.

- `df`: Input DataFrame
- `columns`: Specific columns to process or 'all' for all string columns
- `return_mappings`: Whether to return mapping dictionaries

#### `convert_float_to_int(df, columns='all')`
Converts float columns to integers when all values are whole numbers.

#### `handle_nans(df)`
Handles missing values based on column data types:
- Float columns: Replace with mean
- Integer columns: Replace with max+1

#### `preprocess_df(df, mapping=False)`
Main preprocessing function that combines string mapping, type conversion, and NaN handling.

#### `encode(df, mappings)`
Strictly encodes DataFrame using provided mappings.

#### `decode(encoded_df, mappings)`
Decodes DataFrame using provided mappings (inverse of encode).

#### `remove_classification_anomalies(df, y, contamination=0.05, random_state=42)`
Removes anomalies using Isolation Forest per class.

#### `solve(model, x_cols, y_col, pred_col_name, ...)`
Complete pipeline function that handles:
- Data loading
- Preprocessing
- Anomaly removal
- Model training
- Prediction
- Submission generation

### Utility Functions

#### `nans_look(df)`
Visualizes missing values in the DataFrame.

#### `corr_matrix(df)`
Returns a styled correlation matrix for easy visualization.

#### `preprocess_train_test(train_df, test_df)`
Preprocesses training and test data consistently.

## Examples

### Data Visualization

```python
from ezml import nans_look, corr_matrix

# Visualize missing values
nans_look(df)

# View correlation matrix
corr_matrix(df)
```

### Custom Preprocessing

```python
from ezml import map_unique_strings, convert_float_to_int, handle_nans

# Step-by-step processing
df_processed = map_unique_strings(df, columns=['category1', 'category2'])
df_processed = convert_float_to_int(df_processed)
df_processed = handle_nans(df_processed)
```

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yellowbrick

## License

This project has no license yet.


## Russian

# EZML - Библиотека для упрощения предобработки данных и машинного обучения

Набор инструментов на Python для упрощения предобработки данных и организации рабочих процессов машинного обучения.

## Возможности

- **Кодирование строк**: Автоматическое преобразование строковых значений в числовые
- **Оптимизация типов данных**: Преобразование подходящих float-столбцов в целочисленные
- **Обработка пропущенных значений**: Интеллектуальная обработка NaN на основе типов данных
- **Обнаружение аномалий**: Удаление выбросов с помощью Isolation Forest
- **Кодирование/декодирование**: Согласованное преобразование между обучающими и тестовыми наборами
- **Визуализация**: Графическое представление пропущенных значений
- **Полный пайплайн**: Комплексное решение от загрузки данных до генерации результатов

## Установка

```bash
pip install pandas numpy scikit-learn yellowbrick matplotlib seaborn
```

Просто скопируйте файл `ezml.py` в директорию вашего проекта.

## Использование

### Базовая предобработка

```python
import pandas as pd
from ezml import preprocess_df

# Загрузка данных
df = pd.read_csv('data.csv')

# Предобработка одним вызовом функции
processed_df = preprocess_df(df)
```

### Расширенное использование с сохранением преобразований

```python
# Получение обработанных данных вместе со словарями преобразований
processed_df, mappings = preprocess_df(df, mapping=True)

# Кодирование новых данных с использованием существующих преобразований
new_data_encoded = encode(new_data, mappings)

# Декодирование обратно в исходные значения
original_data = decode(encoded_data, mappings)
```

### Полный ML пайплайн

```python
from sklearn.ensemble import RandomForestClassifier
from ezml import solve

# Настройка и запуск полного пайплайна
model = RandomForestClassifier()
x_cols = ['feature1', 'feature2', 'feature3']
y_col = 'target'

submission = solve(
    model=model,
    x_cols=x_cols,
    y_col=y_col,
    pred_col_name='predictions',
    train_df_filepath='train.csv',
    test_df_filepath='test.csv',
    submission_path='submission.csv',
    id_test_col_name='id',
    id_submission_col_name='id'
)
```

## Справочник API

### Основные функции

#### `map_unique_strings(df, columns='all', return_mappings=False)`
Преобразует строковые столбцы в числовые представления.

- `df`: Входной DataFrame
- `columns`: Конкретные столбцы для обработки или 'all' для всех строковых столбцов
- `return_mappings`: Возвращать ли словари преобразований

#### `convert_float_to_int(df, columns='all')`
Преобразует float-столбцы в целочисленные, когда все значения являются целыми числами.

#### `handle_nans(df)`
Обрабатывает пропущенные значения на основе типов данных столбцов:
- Float-столбцы: Замена на среднее значение
- Integer-столбцы: Замена на (максимальное значение + 1)

#### `preprocess_df(df, mapping=False)`
Основная функция предобработки, объединяющая преобразование строк, конвертацию типов и обработку NaN.

#### `encode(df, mappings)`
Строгое кодирование DataFrame с использованием предоставленных преобразований.

#### `decode(encoded_df, mappings)`
Декодирование DataFrame с использованием предоставленных преобразований (обратное к encode).

#### `remove_classification_anomalies(df, y, contamination=0.05, random_state=42)`
Удаляет аномалии с использованием Isolation Forest для каждого класса.

#### `solve(model, x_cols, y_col, pred_col_name, ...)`
Функция полного пайплайна, которая обрабатывает:
- Загрузку данных
- Предобработку
- Удаление аномалий
- Обучение модели
- Предсказание
- Генерацию результатов

### Вспомогательные функции

#### `nans_look(df)`
Визуализирует пропущенные значения в DataFrame.

#### `corr_matrix(df)`
Возвращает стилизованную матрицу корреляций для удобной визуализации.

#### `preprocess_train_test(train_df, test_df)`
Согласованная предобработка обучающих и тестовых данных.

## Примеры

### Визуализация данных

```python
from ezml import nans_look, corr_matrix

# Визуализация пропущенных значений
nans_look(df)

# Просмотр матрицы корреляций
corr_matrix(df)
```

### Кастомная предобработка

```python
from ezml import map_unique_strings, convert_float_to_int, handle_nans

# Пошаговая обработка
df_processed = map_unique_strings(df, columns=['category1', 'category2'])
df_processed = convert_float_to_int(df_processed)
df_processed = handle_nans(df_processed)
```

## Требования

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yellowbrick

## Лицензия

Этот проект является открытым и доступен по лицензии MIT.
