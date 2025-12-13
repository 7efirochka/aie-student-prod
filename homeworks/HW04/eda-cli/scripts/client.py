import requests
import json

print("Вариант Е. Небольшой скрипт для сервиса.")
print("Скрипт для quality-from-csv :")
print("\n")
url_1 = "http://localhost:8000/quality-from-csv"

try:
    with open("../data/example.csv", "rb") as f:
        files = {'file': ("example.csv", f, "text/csv")}
        response = requests.post(url_1, files = files)
        data = response.json()

        print(f"Валидность данных: {data['ok_for_model']}")
        print(f"Интегральный коэфициент: {data['quality_score']}")  
        print(f"Время выполнения: {data['latency_ms']}")
        print(f"Флаги:")
        for i in data["flags"]:
            print(f'{i}: {data["flags"][i]}')
        print(f"Размеры датафрейма: {data['dataset_shape']['n_rows']} строк, {data['dataset_shape']['n_cols']} столбцов")

except requests.exceptions.ConnectionError:
    print("Ошибка подключения")

finally:
    f.close()

print("""
---------------------------------------------------------------------
      """)

print("Скрипт для summary-from-csv :")
print("\n")

url_2 = "http://localhost:8000/summary-from-csv"

try:
    with open("../data/example_2.csv", "rb") as f:
        files = {'file': ("example_2.csv", f, "text/csv")}
        response = requests.post(url_2, files = files)
        data = response.json()

        print(f"Интегральный коэфициент: {data['quality_score']['quality_score']}")  
        print(f"Размеры датафрейма: {data['dataset_info']['n_rows']} строк, {data['dataset_info']['n_columns']} столбцов")

        print(f"Проблемные столбцы: ", end = " ")
        if data['problematic_column']['problematic_column']:
            for i in data['problematic_column']['problematic_column']:
                print(i, end = " ")
            print("\n")

except requests.exceptions.ConnectionError:
    print("Ошибка подключения")

finally:
    f.close()
