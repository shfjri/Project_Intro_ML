# Implementasi Algoritma K-Nearest Neighbors Untuk Identifikasi Radioisotop

Muhammad Sholih Fajri


## Background

Berbicara tentang machine learning, k-Nearest Neighbors (kNN) adalah algoritma machine learning classifier populer yang paling sederhana. kNN pertama kali diperkenalkan oleh T. Cover dan P. Hart pada tahun 1967 dimana algoritma ini mengklasifikasikan kelas sampel berdasarkan kelas tetangga terdekatnya [1]. kNN sering disebut juga sebagai lazy learner, karena kNN mempelajari dan mengklasifikasi data tanpa membangun sebuah model. Tidak seperti algoritma klasifikasi berbasis model, classifier kNN hanya perlu mengingat semua data training dalam memori [2].

Data spektroskopi gamma biasa digunakan sebagai media untuk mempelajari sifat dari sumber radiasi gamma seperti identifikasi isotop dan estimasi aktivitas radioaktif. Untuk spektroskopi gamma beresolusi tinggi, data dapat dianalisis menggunakan metode tradisional seperti fitting dan identifikasi puncak. 

Berdasarkan hal tersebut di atas, saya mencoba untuk mengimplementasikan algoritma kNN untuk mengklasifikasi sebuah radioisotop berdasarkan data spektroskopi gamma resolusi rendah. Harapannya setelah mengimplementasikan algoritma kNN untuk identifikasi radioisotop, dapat diketahui apakah algoritma kNN cukup dapat diandalkan dalam mengidentifikasi radioisotop atau tidak.

## Related Work

Pustaka khusus yang didedikasikan untuk mempelajari data spektroskopi gamma beresolusi tinggi telah tersedia [4]. Namun, metode ini tidak cukup baik ketika digunakan pada data gamma spektroskopi beresolusi rendah. Beberapa studi menunjukkan bahwa algoritma machine learning seperti neural network (NN) [3-4] dan support vector machine (SVM) [5] sangat efektif untuk mempelajari data spektroskopi gamma beresolusi rendah. Sementara pada penelitian lain sebuah Deep Neural Network digunakan untuk mengidentifikasi beberapa radioisotop dengan data hasil simulasi dan memiliki hasil yang dapat dikatakan _goodfit_ [6].

Penelitian-penelitian tersebut menggunakan data spektroskopi gamma dengan resolusi yang cukup tinggi. Pada penelitian [6] menggunakan data dengan 4096 channel (nilai energi), penelitian [3-4] menggunakan data dengan 1024 channel dan penelitian [5] menggunakan data dengan 498 channel. Sementara data spektroskopi gamma dengan resolusi yang lebih rendah yaitu 256 channel. Dengan perbedaan tersebut, menarik untuk melihat apakah data spektroskopi gamma resolusi rendah dengan 256 channel dapat dipelajar dengan baik oleh algoritma machine learning yang sederhana seperti kNN untuk mengklasifikasi radioisotopnya.

## Dataset and Features

Dataset yang digunakan diambil langsung di laboratorium fisika UIN Jakarta pada tahun 2019. Dataset ini diambil dengan memanfaatkan sintilator NaI untuk mendeteksi jumlah cacahan pada tiap channel energi dari tiap radioisotop yang digunakan. Radioisotop yang digunakan ada 5, yaitu Co<sub>60</sub>, Na<sub>22</sub>, Am<sub>241</sub>, Cs<sub>137</sub> dan Sr<sub>90</sub>. Pada saat pengambilan data, dilakukan variasi waktu cacahan yaitu 60, 120 dan 180 detik untuk data _training_ dan _validation_. Sementara untuk data _tesing_ dilakukan variasi waktu cacahan 45, 90, 150 dan 240 detik.
Berikut adalah gambaran cara pengambilan data:

![praktik](https://github.com/shfjri/Project_Intro_ML/blob/master/images/praktik.png)

Dataset yang digunkanan terdiri dari _training_ dan _validation_ dengan total data sebanyak 1000 (800 _training_ dan 200 _validation_) data serta data _testing_ sebanyak 100 data. Fitur-fitur pada tiap dataset adalah channel energi yang ditangkap oleh sintilator dan jumlah cacahannya atau intensitasnya. Berikut adalah contoh dari dataset yang digunakan:

![sample_dataset](https://github.com/shfjri/Project_Intro_ML/blob/master/images/sample_dataset.png)

Pada dataset, dilakukan scaling dengan `StandardScaler` dari `scikit-learn` dan juga diterapkan `Principal Component Analysis` (`PCA`) untuk mereduksi dimensi dari data. `PCA` sendiri dilakukan untuk membandingkan performa kNN apakah dengan fitur yang lebih sedikit, performa kNN akan lebih buruk, sama atau lebih baik daripada menggunakan semua 256 fitur data.

## Method

Seperti yang sudah disebutkan di atas, metode atau algoritma yang digunakan untuk mengklasifikasi radioisotop dengan data spektroskopi gamma rendah adalah kNN. kNN menggunakan konsep jarak untuk mencari tetangga-tetangga terdekat dari sebuah input data. Untuk menghitung jarak ke tiap tetangga, bisa menggunakan beberapa jenis jarak seperti `Manhattan`, `Euclidean` dan `Minkowski`.
Berikut adalah perbedaan dari ketiga jenis jarak tersebut:

![distance](https://github.com/shfjri/Project_Intro_ML/blob/master/images/distance.png)

Dari sekian tetangga-tetangga terdekat tersebut atau biasa disebut dengan _k_ tetangga terdekat, input data tersebut akan diklasifikasikan mengikuti kelas yg mayoritas dari _k_ tetangga terdekat tersebut.
Secara rinci, langkah-langkah algoritma kNN dalam menentukan sebuah data masuk ke kelas tertentu adalah sebagai berikut:
1. Tentukan jumlah tetangga terdekat (_k_)
2. Hitung jarak data input dengan semua data
3. Pilih _k_ data terdekat dari data input
4. Lakukan _majority vote_ untuk menentukan kelas dari data input

## Results

Algoritma kNN dengan _hyperparameter_ `metric=manhattan` dan `n_neighbors=3` adalah model terbaik untuk menyelesaikan kasus klasifikasi radioisotop dengan data spektroskopi gamma resolusi rendah. _Hyperparameter_ tersebut didapat dengan menggunakan `GridSearchCV` dari `scikit-learn`. Ketika melakukan pencarian model terbaik dengan `GridSearchCV`, _input hyperparameter_ yang dicoba adalah `metric=[manhattan, euclidean, minkowski]` dan `n_neighbors=[3,5,7,9,11]`.

Model kNN yang digunakan memiliki score yang cukup baik pada data _training_. Pada data yang menggunakan semua 256 fitur channel energi, model kNN memiliki score pada data _training_ sebesar 0.99125.

Begitu pula ketika dites untuk memprediksi kelas dari data _validation_ hasilnya cukup baik. Meskipun masih terdapat 3 data yang sebenarnya adalah  `Na` (Na<sub>22</sub>) namun diprediksi sebagai `Sr` (Sr<sub>90</sub>). Hal itu kemungkinan karena fitur data pada kelas `Na` (Na<sub>22</sub>) dan `Sr` (Sr<sub>90</sub>) memiliki sedikit kemiripan. Berikut adalah confusion matrix untuk hasil prediksi pada data _validation_:

![confusion matrix validation](https://github.com/shfjri/Project_Intro_ML/blob/master/images/conf_matrix_val.png)

Jika melihat nilai _precision_, _recall_, _f1-score_ serta _accuracy_ pada data _validation_ didapat hasil yang sangat baik. _Precision_ sebesar 0.99, _recall_ sebesar 0.98_, _f1-score_ ataupun _accuracy_ sebesar 0.98.

![score_validation](https://github.com/shfjri/Project_Intro_ML/blob/master/images/score_val.png)

Hasil prediksi pada data _test_ juga menunjukkan hasil yang cukup baik seperti pada data _validation_ meskipun data _test_ ini berbeda waktu cacahannya dibandingkan dengan data _training_ dan _validation_. Hasil prediksi dan _score_ pada data _test_ dapat dilihat sebagai berikut:

![confusion_matrix_test](https://github.com/shfjri/Project_Intro_ML/blob/master/images/conf_matrix_test.png)
![score_test](https://github.com/shfjri/Project_Intro_ML/blob/master/images/score_test.png)

Terdapat 2 data yang salah diprediksi yaitu 1 data yang sebenarnya adalah `Co` (Co<sub>60</sub>) namun diprediksi sebagai `Sr` (Sr<sub>90</sub>) dan 1 data `Na` (Na<sub>22</sub>) yang diprediksi sebagai `Sr` (Sr<sub>90</sub>). Sementara nilai _precision_, _recall_, _f1-score_ dan _accuracy_ pada data test ini adalah 0.98 untuk _metric-metric_ tersebut.

Melihat hasil tersebut yang menggunakan semua fitur data, mungkin akan berbeda hasilnya jika hanya menggunakan beberapa fitur data saja hasil `PCA`. `PCA` dilakukan pada data _training_, _validation_ dan _test_ dengan hanya akan menggunakan 4 _principal component_ saja sebagai input. Untuk data yang diterapkan `PCA` ini dilakukan hal yang sama dalam mencari model kNN terbaik untuk data hasil `PCA` tersebut. Kemudian setelah didapatkan model kNN dengan _hyperparameters_ terbaik, model digunakan untuk memprediksi kelas pada data _validation_ dan _test_ hasil `PCA`.
_Score_ model kNN untuk data _train_ hasil `PCA` memiliki nilai yang lebih tinggi dibanding data _train_ dengan semua fitur, yaitu 1.0.

Jika dilihat pada _confusion matrix_ di bawah ini, terlihat bahwa ada perbaikan hasil prediksi pada data _validation_ yang diterapkan `PCA`. Tidak ada data yang salah diprediksi. Ini akan menghasilkan nilai _precision_, _recall_, _f1-score_ dan _accuracy_ yang lebih tinggi. Semua _metric_ tersebut memiliki nilai sebesar 1.0.

![confusion_matrix_val_pca](https://github.com/shfjri/Project_Intro_ML/blob/master/images/conf_matrix_val_pca.png)
![score_val_pca](https://github.com/shfjri/Project_Intro_ML/blob/master/images/score_val_pca.png)

Ketika model memprediksi data _test_ yang diterapkan `PCA`, hasilnya tidak sebaik seperti pada data _validation_ di atas. Terdapat 1 data yang sebenarnya adalah `Co` (Co<sub>60</sub>) diprediksi sebagai `Sr` (Sr<sub>90</sub>). Namun ini masih lebih baik dibandingkan dengan data _test_ yang menggunakan semua fitur data. Begitu pula untuk nilai _precison_, _recall_, _f1-score_ dan _accuracy_ nya sebesar 0.99.
Berikut adalah _confusion matrix_ dan _score_ _metric-metric_ tersebut:

![confusion_matrix_test_pca](https://github.com/shfjri/Project_Intro_ML/blob/master/images/conf_matrix_test_pca.png)
![score_test_pca](https://github.com/shfjri/Project_Intro_ML/blob/master/images/score_test_pca.png)

Dengan hasil yang lebih baik jika melakukan `PCA` pada data, maka waktu komputasi bisa dihemat, dimensi data bisa jauh berkurang dan akan lebih mudah untuk memvisualisasikan hubungan antar fitur. Berikut adalah contoh hubungan antar fitur _principal component_ yang bisa membedakan tiap-tiap kelas dengan cukup baik:

![scatter_plot_pc1_pc3](https://github.com/shfjri/Project_Intro_ML/blob/master/images/scatter_pc1_pc3.png)

Dapat dilihat bahwa dengan _principal component 1_ (_PC1_) dan _principal component 3_ (_PC3_), data-data tiap kelas dapat dibedakan dengan cukup baik. Meskipun terlihat antara `Na` (Na<sub>22</sub>), `Am` (Am<sub>241</sub>) dan `Sr` (Sr<sub>90</sub>) terdapat data-data yang berdekatan, punya fitur _PC1_ dan _PC3_ yang mirip.

## Conclusion

Berdasarkan _score_ pada data _training_, _validation_ dan _test_, baik yang menggunakan semua fitur data atau yang diterapkan `PCA` memiliki nilai yang cukup tinggi di semau set data, algoritma kNN dapat dikatakan __goodfit__ untuk melakukan klasifikasi radioisotop dengan input data berupa data spektroskopi gamma resolusi rendah. Mengingat juga algoritma kNN ini hanya mengandalkan memori untuk mengingat tetangga-tetangga terdekat dalam menentukan kelas dari sebuah data, dengan model yang sangat sederhana ini permasalahan klasifikasi radioisotop berdasarkan data spektroskopi gamma resolusi rendah dapat dilakukan.
Hal ini diharapkan menjadi pemantik untuk mencoba algoritma kNN untuk melakukan hal yang sama namun pada data spektroskopi gamma resolusi tinggi.

## References

<ul> [1] T.Cover and P. Hart, “Nearest neighbor pattern classification,” IEEE Trans. Inf. Theory, vol. 13, no. 1, pp. 21–27, 1967. </ul>
<ul> [2] S. Zhang, “Cost-sensitive KNN classification,” Neurocomputing, vol. 391, pp. 234–242, 2020. </ul>
<ul> [3] M. Kamuda and C. J. Sullivan, “An automated isotope identification and quantification algorithm for isotope mixtures in low-resolution gamma-ray spectra,” Radiat. Phys. Chem., vol. 155, no. June 2018,
pp. 281–286, 2019. </ul>
<ul> [4] M. Kamuda, J. Zhao, and K. Huff, “A comparison of machine learning methods for automated gamma-ray spectroscopy,” Nucl. Instruments Methods Phys. Res. Sect. A Accel. Spectrometers, Detect.
Assoc. Equip., vol. 954, no. October, 2020. </ul>
<ul> [5] H. Hata, K. Yokoyama, Y. Ishimori, Y. Ohara, Y. Tanaka, and N. Sugitsue, “Application of support vector machine to rapid classification of uranium waste drums using low-resolution γ-ray spectra,”
Appl. Radiat. Isot., vol. 104, pp. 143–146, 2015. </ul>
<ul> [6] T. Kin, J. Goto and M. Oshima, "Machine learning approach for gamma-ray spectra identification for radioactivity analysis," IEEE Nuclear Science Symposium and Medical Imaging Conference (NSS/MIC), 2019. </ul>
