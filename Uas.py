import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import normaltest
from scipy.stats import kstest, norm
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Walmart.csv', usecols=['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
class Num1() :
    print('Number 1')
    print('\n')
    
    
    print('Num 1b')
    store_id = 4
    df_filter = df[df['Store'] == store_id]
    
    mean_w = df_filter['Holiday_Flag'].mean()
    mean_h = df_filter['Weekly_Sales'].mean()
    mean_t = df_filter['Temperature'].mean()
    mean_f = df_filter['Fuel_Price'].mean()
    mean_c = df_filter['CPI'].mean()
    mean_u = df_filter['Unemployment'].mean()

    median_w = df_filter['Holiday_Flag'].median()
    median_h = df_filter['Weekly_Sales'].median()
    median_t = df_filter['Temperature'].median()
    median_f = df_filter['Fuel_Price'].median()
    median_c = df_filter['CPI'].median()
    median_u = df_filter['Unemployment'].median()

    std_w = df_filter['Holiday_Flag'].std()
    std_h = df_filter['Weekly_Sales'].std()
    std_t = df_filter['Temperature'].std()
    std_f = df_filter['Fuel_Price'].std()
    std_c = df_filter['CPI'].std()
    std_u = df_filter['Unemployment'].std()


    var_w = df_filter['Holiday_Flag'].var()
    var_h = df_filter['Weekly_Sales'].var()
    variance_t = df_filter['Temperature'].var()
    variance_f = df_filter['Fuel_Price'].var()
    variance_c = df_filter['CPI'].var()
    variance_u = df_filter['Unemployment'].var()
    
    print("\n")
    print("Statistik untuk STORE 4: ")
    print("\n")
    print("Weekly Sale: ")
    print("Mean: ", mean_w)
    print("Median: ", median_w)
    print("Simpangan Baku: ", std_w)
    print("Varians: ", var_w, "\n")

    print("Holiday_Flag: ")
    print("Mean:", mean_h)
    print("Median:", median_h)
    print("Simpangan Baku:", std_h)
    print("Varians:", var_h, "\n")

    print("Temperature: ")
    print("Mean:", mean_t)
    print("Median:", median_t)
    print("Simpangan Baku:", std_t)
    print("Varians:", variance_t, "\n")

    print("Fuel Price: ")
    print("Mean:", mean_f)
    print("Median:", median_f)
    print("Simpangan Baku:", std_f)
    print("Varians:", variance_f, "\n")


    print("CPI: ")
    print("Mean:", mean_c)
    print("Median:", median_c)
    print("Simpangan Baku:", std_c)
    print("Varians:", variance_c, "\n")
    
    print("Unemployment: ")
    print("Mean:", mean_u)
    print("Median:", median_u)
    print("Simpangan Baku:", std_u)
    print("Varians:", variance_u, "\n")
    
    print('\n')
    print('Num 1c')
    fuel_price_q1 = df_filter['Fuel_Price'].quantile(0.25)
    fuel_price_q2 = df_filter['Fuel_Price'].quantile(0.50)
    fuel_price_q3 = df_filter['Fuel_Price'].quantile(0.75)
    fuel_price_iqr = fuel_price_q3 - fuel_price_q1

    cpi_q1 = df_filter['CPI'].quantile(0.25)
    cpi_q2 = df_filter['CPI'].quantile(0.50)
    cpi_q3 = df_filter['CPI'].quantile(0.75)
    cpi_iqr = cpi_q3 - cpi_q1

    unemployment_q1 = df_filter['Unemployment'].quantile(0.25)
    unemployment_q2 = df_filter['Unemployment'].quantile(0.50)
    unemployment_q3 = df_filter['Unemployment'].quantile(0.75)
    unemployment_iqr = unemployment_q3 - unemployment_q1

    print("Nilai Q1, Q2, Q3, dan IQR untuk 'Fuel_Price', 'CPI', dan 'Unemployment' untuk STORE 4")
    print("Fuel Price:")
    print("Q1:", fuel_price_q1)
    print("Q2:", fuel_price_q2)
    print("Q3:", fuel_price_q3)
    print("IQR:", fuel_price_iqr)

    print("CPI:")
    print("Q1:", cpi_q1)
    print("Q2:", cpi_q2)
    print("Q3:", cpi_q3)
    print("IQR:", cpi_iqr)

    print("Unemployment:")
    print("Q1:", unemployment_q1)
    print("Q2:", unemployment_q2)
    print("Q3:", unemployment_q3)
    print("IQR:", unemployment_iqr)
    
    print("\n")
    print('Num 1d')
    print("\n")
    group_data = df.groupby('Holiday_Flag')['Weekly_Sales'].var()
    print("Variance Description:")
    for flag, variance in group_data.items():
        if flag == 1:
            print("Holiday Week:")
        else:
            print("Non-Holiday Week:")
        print("Variance:", variance)
    
    print("\n")
    print('Num 1e')
    print("\n")
    average_sales_by_store = df.groupby('Store')['Weekly_Sales'].mean()
    is_average_sales_equal = average_sales_by_store.nunique() == 1
    if is_average_sales_equal:
        print("Rata-rata Weekly Sales di setiap toko sama.")
    else:
        print("Rata-rata Weekly Sales di setiap toko berbeda.")
    
    print("\n")
    print('Num 1f')
    print("\n")
    max_cpi_by_store = df.groupby('Store')['CPI'].max()
    higher_cpi_by_store = max_cpi_by_store.idxmax()
    higher_cpi_value = max_cpi_by_store.max()

    print("Store dengan CPI paling tinggi:")
    print("Store:", higher_cpi_by_store)
    print("Nilai CPI tertinggi:", higher_cpi_value)
        
    print("\n")
    print('Num 1g')
    print("\n")
    average_cpi_holiday = df[df['Holiday_Flag'] == 1]['CPI'].mean()
    average_cpi_non_holiday = df[df['Holiday_Flag'] == 0]['CPI'].mean()

    if average_cpi_holiday > average_cpi_non_holiday:
        print("Rata-rata CPI pada holiday week lebih tinggi.")
    elif average_cpi_holiday < average_cpi_non_holiday:
        print("Rata-rata CPI pada non-holiday week lebih tinggi.")
    else:
        print("Rata-rata CPI pada holiday week dan non-holiday week sama.")
    
class Num2() :
    weekly_sales = df['Weekly_Sales']


    fuel_price = df['Fuel_Price']
    alpha = 0.05

    print("\n")
    print('Num 2a')
    print("\n")

    statistic, p_value = kstest(weekly_sales, 'norm', norm.fit(weekly_sales))
    print("Uji Normalitas Weekly Sales:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
    if p_value > alpha:
        print("Weekly Sales didistribusikan secara normal")
    else:
        print("Weekly Sales tidak didistribusikan secara normal")

    statistic, p_value = kstest(fuel_price, 'norm', norm.fit(fuel_price))
    print("Uji Normalitas Fuel Price:")
    print(f"Statistic: {statistic}")
    print(f"P-value: {p_value}")
    if p_value > alpha:
        print("Fuel Price didistribusikan secara normal")
    else:
        print("Fuel Price tidak didistribusikan secara normal")
        
        
class Num3():
    print("\n")
    print('Num 3a')
    print("\n")
    correlation = df[['Holiday_Flag', 'Temperature',
                      'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
    print("Nilai korelasi antara variabel independen dan variabel dependen:")
    print(correlation['Weekly_Sales'])

    print("\n")
    print('Num 3b')
    print("\n")
    correlation = df[['Holiday_Flag', 'Temperature',
                      'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
    negative_correlations = correlation[correlation['Weekly_Sales'] < 0]
    negative_correlations = negative_correlations['Weekly_Sales'].drop(
        'Weekly_Sales', errors='ignore')
    if negative_correlations.empty:
        print(
            "Tidak ada pasangan variabel independen dan dependen dengan korelasi negatif.")
    else:
        print("Pasangan variabel independen dan dependen dengan korelasi negatif:")
        print(negative_correlations)

class Num4() :
    X = df[['Fuel_Price']]


    y = df['Weekly_Sales']

    # Inisialisasi model regresi linear
    model = LinearRegression()

    # Melatih model menggunakan data
    model.fit(X, y)

    # Prediksi nilai y berdasarkan X
    y_pred = model.predict(X)

    # Menampilkan scatter plot data
    plt.scatter(X, y, color='blue', label='Data')

    # Menampilkan garis regresi
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

    # Menampilkan label dan judul pada grafik
    plt.xlabel('Fuel_Price')
    plt.ylabel('Weekly_Sales')
    plt.title('Linear Regression')

    # Menampilkan legenda
    plt.legend()

    # Menampilkan grafik
    plt.show()
    
    

