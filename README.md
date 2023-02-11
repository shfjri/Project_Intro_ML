# Implementasi Algoritma K-Nearest Neighbors Untuk Identifikasi Radioisotop

Muhammad Sholih Fajri


## Background

Berbicara tentang machine learning, k-Nearest Neighbors (kNN) adalah algoritma machine learning classifier populer yang paling sederhana. kNN pertama kali diperkenalkan oleh T. Cover dan P. Hart pada tahun 1967 dimana algoritma ini mengklasifikasikan kelas sampel berdasarkan kelas tetangga terdekatnya [1]. kNN sering disebut juga sebagai lazy learner, karena kNN mempelajari dan mengklasifikasi data tanpa membangun sebuah model. Tidak seperti algoritma klasifikasi berbasis model, classifier kNN hanya perlu mengingat semua data training dalam memori [2].

Data spektroskopi gamma biasa digunakan sebagai media untuk mempelajari sifat dari sumber radiasi gamma seperti identifikasi isotop dan estimasi aktivitas radioaktif. Untuk spektroskopi gamma beresolusi tinggi, data dapat dianalisis menggunakan metode tradisional seperti fitting dan identifikasi puncak. 

Berdasarkan hal tersebut di atas, saya mencoba untuk mengimplementasikan algoritma kNN untuk mengklasifikasi sebuah radioisotop berdasarkan data spektroskopi gamma resolusi rendah. Harapannya setelah mengimplementasikan algoritma kNN untuk identifikasi radioisotop, dapat diketahui apakah algoritma kNN cukup dapat diandalkan dalam mengidentifikasi radioisotop atau tidak.

## Related Work

Pustaka khusus yang didedikasikan untuk mempelajari data spektroskopi gamma beresolusi tinggi telah tersedia [16]. Namun, metode ini tidak cukup baik ketika digunakan pada data gamma spektroskopi beresolusi rendah. Beberapa studi menunjukkan bahwa algoritma machine learning seperti neural network (NN) [15-16] dan support vector machine (SVM) [17] sangat efektif untuk mempelajari data spektroskopi gamma beresolusi rendah. Sementara pada penelitian lain sebuah Deep Neural Network digunakan untuk mengidentifikasi beberapa radioisotop dengan data hasil simulasi dan memiliki hasil yang dapat dikatakan _goodfit_ [T. Kin dkk].

Penelitian-penelitian tersebut menggunakan data spektroskopi gamma dengan resolusi yang cukup tinggi. Pada penelitian [T. Kin dkk] menggunakan data dengan 4096 channel (nilai energi), penelitian [15-16] menggunakan data dengan 1024 channel. Sementara data spektroskopi gamma yang saya gunakan hanya 256 channel. Dengan perbedaan tersebut, menarik untuk melihat apakah data spektroskopi gamma resolusi rendah dengan 256 channel dapat dipelajar dengan baik oleh algoritma machine learning yang sederhana seperti kNN untuk mengklasifikasi radioisotopnya.

## Dataset and Features

Dataset yang digunakan diambil langsung di laboratorium fisika UIN Jakarta pada tahun 2019. Dataset ini diambil dengan memanfaatkan sintilator NaI untuk mendeteksi jumlah cacahan pada tiap channel energi dari tiap radioisotop yang digunakan. Radioisotop yang digunakan ada 5, yaitu Co<sub>60</sub>, Na<sub>22</sub>, Am<sub>241</sub>, Cs<sub>137</sub> dan Sr<sub>90</sub>. Pada saat pengambilan data, dilakukan variasi waktu cacahan yaitu 60, 120 dan 180 detik untuk data _training_ dan _validation_. Sementara untuk data _tesing_ dilakukan variasi waktu cacahan 45, 90, 150 dan 240 detik.
Berikut adalah gambaran cara pengambilan data:

![praktik](https://drive.google.com/file/d/1M7-UmQe3HX7adP3Tsb9H70cG52FT5OIH/view?usp=sharing)

Dataset yang digunkanan terdiri dari _training_ dan _validation_ dengan total data sebanyak 1000 (800 _training_ dan 200 _validation_) data serta data _testing_ sebanyak 100 data. Fitur-fitur pada tiap dataset adalah channel energi yang ditangkap oleh sintilator dan jumlah cacahannya atau intensitasnya. Berikut adalah contoh dari dataset yang digunakan:

![sample_dataset](https://drive.google.com/file/d/1rhDOTgKINikJqJ0CxM0zzTy6JT5HF0kS/view?usp=share_link)

Pada dataset, dilakukan scaling dengan `StandardScaler` dari `scikit-learn` dan juga diterapkan `Principal Component Analysis` (`PCA`) untuk mereduksi dimensi dari data. `PCA` sendiri dilakukan untuk membandingkan performa kNN apakah dengan fitur yang lebih sedikit, performa kNN akan lebih buruk, sama atau lebih baik daripada menggunakan semua 256 fitur data.

## Metode

Seperti yang sudah disebutkan di atas, metode atau algoritma yang digunakan untuk mengklasifikasi radioisotop dengan data spektroskopi gamma rendah adalah kNN. kNN menggunakan konsep jarak untuk mencari tetangga-tetangga terdekat dari sebuah input data. Untuk menghitung jarak ke tiap tetangga, bisa menggunakan beberapa jenis jarak seperti `Manhattan`, `Euclidean` dan `Minkowski`.
Berikut adalah perbedaan dari ketiga jenis jarak tersebut:

![distance](https://drive.google.com/file/d/1xGY9mNtkwSFbAvJ6TC0nG6HGwg2xGPWH/view?usp=share_link)

Dari sekian tetangga-tetangga terdekat tersebut atau biasa disebut dengan _k_ tetangga terdekat, input data tersebut akan diklasifikasikan mengikuti kelas yg mayoritas dari _k_ tetangga terdekat tersebut.
Secara rinci, langkah-langkah algoritma kNN dalam menentukan sebuah data masuk ke kelas tertentu adalah sebagai berikut:
1. Tentukan jumlah tetangga terdekat (_k_)
2. Hitung jarak data input dengan semua data
3. Pilih _k_ data terdekat dari data input
4. Lakukan _majority vote_ untuk menentukan kelas dari data input

## Results

Algoritma kNN dengan _hyperparameter_ `metric=manhattan` dan `n_neighbors=3` adalah model terbaik untuk menyelesaikan kasus klasifikasi radioisotop dengan data spektroskopi gamma resolusi rendah. _Hyperparameter_ tersebut didapat dengan menggunakan `GridSearchCV` dari `scikit-learn`. Ketika melakukan pencarian model terbaik dengan `GridSearchCV`, _input hyperparameter_ yang dicoba adalah `metric=[manhattan, euclidean, minkowski]` dan `n_neighbors=[3,5,7,9,11]`.

Model kNN yang digunakan memiliki score yang cukup baik pada data _training_, _validation_ dan _testing_. Pada data yang menggunakan semua 256 fitur channel energi, model kNN memiliki score pada data _training_ sebesar 0.99125.

Begitu pula ketika dites untuk memprediksi kelas dari data _validation_ hasilnya cukup baik seperti berikut:

![confusion matrix data validation](https://drive.google.com/file/d/135rrJB_957Mr5foRBtpZdNqonNqIaT67/view?usp=sharing)
