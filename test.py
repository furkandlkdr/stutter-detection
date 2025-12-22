import pandas as pd

# Tüm sütunları gösterecek şekilde ayarla
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# CSV dosyasını oku ve ilk 10 satırını göster
df = pd.read_csv('sep28k-mfcc.csv')
print(df.head(10))