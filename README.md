# lojistik-regresyon-

Bu projede, Telco Customer Churn veri seti kullanılarak Lojistik Regresyon modeli ile müşteri kaybı tahmini yapılmıştır. Veri seti, müşteri bilgileri (yaş, hizmet kullanımı, ödeme yöntemleri gibi) üzerinden, müşterinin hizmeti bırakıp bırakmayacağını tahmin etmeye yöneliktir.

İki model karşılaştırılmıştır:

Scikit-learn Lojistik Regresyon: Scikit-learn kütüphanesi kullanılarak eğitilen standart lojistik regresyon modeli.
Sıfırdan Lojistik Regresyon: Maksimum Likelihood Estimation (MLE) ve Gradient Descent ile sıfırdan yazılmış bir lojistik regresyon modeli.


Değerlendirme Metrikleri

Doğruluk (Accuracy): Modelin doğru tahmin ettiği örneklerin oranı.
Karmaşıklık Matrisi: Modelin sınıf tahminlerinin ayrıntılı analizini sağlar.

Scikit-learn Modeli Performansı
Doğruluk: %80.93
Eğitim Süresi: ~0.46 saniye
Karmaşıklık Matrisi:
[[1389,  150]
 [ 253,  321]]

Sıfırdan Model Performansı
Doğruluk: %80.45
Eğitim Süresi: ~2.63 saniye
Karmaşıklık Matrisi:
[[1362,  177]
 [ 236,  338]]

Karmaşıklık Matrisi
Karmaşıklık matrisi, modelin doğru ve yanlış tahminlerini görsel olarak sunar. Her iki model de yüksek doğru pozitif ve doğru negatif tahminleri yapmıştır. Ancak, yanlış negatif ve yanlış pozitif tahminler de mevcuttur.

Sınıf Dağılımının Değerlendirme Üzerindeki Etkisi
Veri setinde sınıflar (churn/no churn) dengeli olduğundan doğruluk iyi bir performans göstergesidir. Ancak, sınıflar arasında büyük bir dengesizlik olsaydı, precision ve recall gibi metrikler daha fazla önem kazanırdı.

Sonuç
Her iki model de müşteri kaybı tahmininde başarılı olmuş, ancak Scikit-learn modeli daha hızlı eğitim süresi ve biraz daha yüksek doğruluk elde etmiştir. Sıfırdan model ise daha uzun sürede eğitim almakta ancak yine de iyi bir doğruluk göstermektedir.
