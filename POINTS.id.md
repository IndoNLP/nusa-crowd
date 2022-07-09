# Pedoman poin kontribusi

Untuk dianggap sebagai co-author, diperlukan 10 poin kontribusi.

## Data Loader

Menerapkan data loader apa pun diberikan +3 poin.
Info lebih lanjut dapat ditemukan [di sini](DATALOADER.md).

## Proposal Dataset

Poin dari proposal dataset tergantung pada berbagai faktor:

### Ukuran

Kami dapat memiliki 4 level berbeda: Small, Medium, Large, XL

- Small (S): <1K (+1 poin)
- Medium (M): 1K<=x<10K (+ 2 poin)
- Large (L): 10K<=x<1M (+3 poin)
- Extra Large (XL): >=1M (+4 poin)

x adalah jumlah sampel (train + val + test). Untuk data yang menggunakan evaluasi k-fold, jumlah data yang dihitung hanya salah satu fold.

### Downstream task dan Bahasa Kelangkaan

- Langka / Tidak ada sumber daya: Tidak ada dataset publik pada bahasa / downstream task ini. Dataset ini akan menjadi yang baru untuk bahasa/ downstream task tertentu. (+6 poin)
- Jarang: Ada beberapa sumber dalam bahasa lokal ini, tetapi sangat sulit ditemukan. (+ 3 poin)
- Umum: dataset untuk downstream task & bahasa umum. (+1 poin)


### Kualitas (untuk Dataset berlabel)

- Excellent (E): Dataset berkualitas tinggi, mis. berlabel/tertulis/beranotasi **dan** dievaluasi oleh manusia dengan persetujuan expert annotator. Protokol anotasi didokumentasikan secara menyeluruh dalam makalah. (poin x1.5)

- Good (G): mis. data dihasilkan secara otomatis (yaitu dengan crawling), tetapi diverifikasi oleh manusia. Atau, data dapat diberi label oleh manusia dengan verifikasi minimal/tanpa verifikasi. (poin x1)

- Poor (P): mis. data sepenuhnya dibuat oleh mesin, tanpa verifikasi. (poin x0,5)


### Maksimal Jumlah Kontributor

- Kontribusi untuk dataset baru terbatas pada 2 author utama. Harap tentukan nama author utama yang ingin Anda tambahkan ke PR.
- Kontribusi untuk dataloader yang dihitung hanya pembuat PR.

## Contoh

Mari kita asumsikan analisis sentimen baru untuk salah satu bahasa Papua, terdiri dari 500 kalimat.
Untuk ukuran data, 500 kalimat ini akan dianggap kecil (+1 pts). Meskipun analisis sentimen adalah umum, tetapi bahasa itu sendiri sangat jarang dan kurang terwakili, oleh karena itu kami mendapat +6 poin untuk ini. Terakhir, dengan asumsi data dalam kualitas tinggi, kita akan mendapatkan total (1 + 6) * 1,5 pts = 10,5 pts, dan ini cukup untuk mendapatkan authorship.

Contoh lain, mari kita asumsikan dataset Natural Language Inference (NLI) baru untuk bahasa Jawa. NLI sendiri bukanlah hal baru untuk bahasa Indonesia, dan tersedia sumber daya bahasa Jawa. Namun NLI Jawa adalah yang pertama, sehingga masih tergolong langka (+6 poin). Dengan asumsi dataset berukuran kecil, dengan kualitas yang baik, kita akan mendapatkan total 7 poin. Dengan tambahan, menerapkan data loader untuk dataset ini, kita akan memiliki total 10 poin, dan ini sudah cukup untuk mendapatkan authorship.
