# Pedoman poin kontribusi

Untuk dianggap sebagai co-author, diperlukan 10 poin kontribusi.

> **Catatan**: Tujuan adanya sistem poin kontribusi bukan untuk menghalangi kolaborasi, melainkan untuk mengapresiasi dataset langka dan berkualitas tinggi. 
Oleh karena itu, kami mungkin menurunkan syarat poin untuk mengakomodasi lebih banyak co-author, jika diperlukan.

## Implementasi Data Loader

Menerapkan data loader apa pun diberikan +3 poin, atau sesuai yang ditentukan di Github issue.
Info lebih lanjut dapat ditemukan [di sini](DATALOADER.md).

## Proposal Datasheet di NusaCatalogue

### Proposal Datasheet sebagai Author Dataset
Mencatatkan datasheet dari sebuah dataset akan mendapatkan nilai +2.

Untuk mendukung keterbukaan dataset, untuk dataset yang tertutup dan akan diubah aksesnya menjadi publik, maka akan diberikan nilai tambahan +2.

Untuk mendukung pengembangan dataset bahasa daerah:
- Untuk dataset bahasa Sunda, Jawa, atau Minang, akan diberikan nilai +2
- Untuk dataset dari bahasa daerah lain, akan diberikan nilai +4

Berdasarkan hasil observasi, kami menemukan bahwa terdapat beberapa jenis task NLP yang banyak ditemukan dalam bahasa-bahasa di Indonesia, antara lain: machine translation (MT), language modeling (LM), sentiment analysis (SA), dan named entity recognition (NER). Untuk mendukung pengembangan korpora NLP dalam task lainnya, semua datasheet yang mencakup task selain yang disebutkan diatas, akan diberikan nilai tambahan +2. 

Selain daripada itu, dikarenakan keterbatasan korpora NLP di bahasa-bahasa Indonesia yang menggunakan modality lain (speech-text, image-text, multimodal), datasheets yang mencakup modality selain text akan diberikan nilai tambahan +2. 

Kami menyadari bahwa kualitas dari sebuah dataset sangatlah beragam. Sebagai bentuk keadilan dalam penilaian dataset, bagi dataset yang tidak memenuhi standar kualitas yang ditentukan, akan diberikan pengurangan nilai sebesar 50%. Ketentuan ini berlaku bagi dataset yang dikumpulkan dengan:
- Crawling tanpa dilakukan pengecekan ulang secara manual
- Pelabelan menggunakan mesin atau aturan heuristik, tanpa dilakukan pengecekan ulang secara manual
- Translasi menggunakan mesin dari dataset bahasa lain tanpa adanya pengecekan ulang secara manual

> **Catatan**: Jika terdapat lebih dari 1 Author dalam pembangunan dataset, Author utama dapat menominasikan 1 orang Author untuk mendapatkan nilai kontribusi yang sama.

### Proposal Datasheet dari Dataset orang lain
Mengajukan datasheet dari sebuah dataset yang dibangun orang lain akan mendapatkan nilai +1.

## Listing Private Dataset
Mencatatkan dataset yang tertutup akan mendapatkan nilai +1 untuk setiap dataset yang dicatatkan.
