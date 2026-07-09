# Cellular Tracking RF-CTRV-PF

Bu repository, hücresel sinyal ölçümleri kullanılarak kullanıcı konum takibi yapılmasını amaçlayan bir çalışmayı içermektedir. Çalışmada Random Forest tabanlı bir ölçüm modeli ve CTRV (Constant turn rate and velocity) hareket modeli kullanan Particle Filter yapısı kullanılmıştır.

Bu hareket modelinde durum vektörü x ve y konumu, hız, açı ve açı değişimini içerir. Toplam 5 elemanlıdır.
## Genel Yöntem

Çalışmada izlenen genel işlem adımları şunlardır:

1. Excel dosyasından hücresel ölçüm verilerinin okunması
2. GPS ve hücresel ölçüm verilerinin temizlenmesi
3. Verilerin 1 saniyelik zaman aralıklarına getirilmesi
4. Random Forest ölçüm modelinin eğitilmesi
5. Eğitilen modelin CTRV Particle Filter içinde kullanılması
6. Gerçek rota ile tahmin edilen rotanın karşılaştırılması
7. Konum hatası için CDF grafiğinin oluşturulması

## Kodların Farkı

Çalışmadaki 2 kodun tek bir farkı vardır. Biri Random Forest Modelinin girdisi olarak sadece x ve y kullanırken diğeri x ve y değerlerine ek olarak manuel olarak hesaplanmış sinüs cosinüs ve mesafe bilgisi de içerir

## Kullanılan Hücresel Ölçümler

Random Forest modelinin çıktısı olarak aşağıdaki 8 hücresel parametre kullanılmaktadır:

```text
NR_RSRP_0
NR_RSRQ_0
NR_SINR_0
CQI_NR
NR_TIMING_ADVANCE
NR_PDSCH_MCS_0
NR_PDSCH_MCS_1
PuschTxPower
