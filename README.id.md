# Selamat datang di NusaCrowd!

<h3>95 datasets telah terdaftar di NusaCrowd</h3>

![Dataset claimed](https://progress-bar.dev/81/?title=Datasets%20Claimed%20(77%20Datasets%20Claimed))

<!-- milestone starts -->
![Milestone 1](https://progress-bar.dev/100/?title=Milestone%201%20(30%20Datasets%20Completed))

![Milestone 2](https://progress-bar.dev/95/?title=Milestone%202%20(60%20Datasets%20Completed))

![Milestone 3](https://progress-bar.dev/57/?title=Milestone%203%20(100%20Datasets%20Completed))

![Milestone 4](https://progress-bar.dev/38/?title=Milestone%204%20(150%20Datasets%20Completed))
<!-- milestone ends -->

*Read this README in [English](README.md).*

NLP Indonesia kurang terwakili dalam komunitas riset, dan salah satu alasannya adalah kurangnya akses ke dataset publik ([Aji et al., 2022](https://aclanthology.org/2022.acl-long.500/)). Untuk mengatasi masalah ini, kami memulai
**NusaCrowd**, kolaborasi bersama untuk mengumpulkan dataset NLP untuk bahasa Indonesia. Bantu kami mengumpulkan dan memusatkan dataset NLP Indonesia, dan menjadi rekan penulis makalah penelitian kami yang akan datang.

## Bagaimana cara berkontribusi?

Anda dapat berkontribusi dengan mengajukan **set data NLP yang tidak terdaftar** di [catatan kami](https://indonlp.github.io/nusa-catalogue/). [Cukup isi formulir ini](https://forms.gle/31dMGZik25DPFYFd6), dan kami akan memeriksa dan menyetujui entri Anda.

Kami akan memberikan **poin kontribusi** berdasarkan beberapa faktor, antara lain: **kualitas dataset**, **kelangkaan bahasa**, atau **kelangkaan downstream task**.

Anda juga dapat mengajukan dataset dari pekerjaan Anda yang lampau, yang masih belum terbuka untuk umum. Pada kasus ini, Anda harus membuat dataset Anda terbuka dengan cara meng-uploadnya ke publik, misalnya melalui Github atau Google Drive.

Anda dapat mengirimkan beberapa entri, dan jika total **poin kontribusi** sudah di atas ambang batas, kami akan menyertakan Anda sebagai rekan penulis (Umumnya cukup mengajukan 1-2 dataset). Baca metode penghitungan poin selengkapnya [di sini](POINTS.id.md).

> **Catatan**: Kami tidak mengambil kepemilikan dari dataset yang disubmit. Lihat FAQ di bawah.

## Ada cara lain untuk membantu?

Ya! Selain pengumpulan dataset baru, kami juga memusatkan dataset yang ada dalam satu skema yang memudahkan peneliti untuk menggunakan dataset NLP Indonesia. Anda dapat membantu kami di sana dengan membuat pemuat dataset. Untuk detail lebih lanjut tentang itu, bisa ditemukan [di sini](DATALOADER.md).

Sebagai alternatif, kami juga mendata paper-paper riset NLP di bahasa-bahasa Indonesia yang mana mereka masih belum membuka datasetnya. Kami akan menghubungi para penulis paper-paper tersebut nanti untuk terlibat di NusaCrowd. Lebih lanjut tentang ini ada di [Slack server](https://join.slack.com/t/nusacrowd/shared_invite/zt-1bbvt4och-JkC7tzeL_eUk4UD6tl3kDg) kami.


## FAQ

#### Siapa yang menjadi pemilik dataset yang disubmit?

NusaCrowd **tidak** membuat duplikat atau salinan dari dataset yang disubmit. Maka, pemilik dataset yang disubmit tetap berada di author asli. NusaCrowd hanya sebatas membuat dataloader, yaitu pengunduh file dan pembaca data untuk menyederhanakan dan mengstandarisasi proses pembacaan data. Kami juga hanya mengumpulkan metadata dari dataset yang disubmit untuk ditampilkan di [katalog kami](https://indonlp.github.io/nusa-catalogue/) agar dataset Anda lebih mudah ditemukan!
Sitasi ke pemilik data asli juga disediakan baik di NusaCrowd atau di katalog kami.

#### Bagaimana cara menemukan lisensi yang sesuai untuk dataset saya?

Lisensi untuk dataset tidak selalu jelas. Berikut adalah beberapa strategi yang bisa dicoba dalam pencarian Anda,

* periksa file seperti README atau LICENSE yang mungkin didistribusikan dengan dataset itu sendiri
* periksa halaman web dataset
* periksa makalah penelitian atau publikasi yang mengumumkan rilis dataset
* periksa situs web organisasi yang menyediakan dataset

Jika tidak ada lisensi resmi yang tercantum di mana pun, tetapi Anda menemukan halaman web yang menjelaskan kebijakan penggunaan data umum untuk dataset, Anda dapat kembali menyediakan URL tersebut dalam variabel `_LICENSE`. Jika Anda tidak dapat menemukan informasi lisensi apa pun, harap dicatat di PR Anda dan masukkan `_LICENSE="Unknown"` di script dataset Anda.

#### Bagaimana jika dataset saya belum tersedia untuk umum?

Anda dapat mengunggah dataset Anda secara publik terlebih dahulu, mis. di Github.

#### Bisakah saya membuat PR jika saya punya ide / mengajukan perubahan kode pada repository nusa-crowd?

Jika Anda memiliki ide untuk repositori nusa-crowd, silakan buat `issue` dan mintalah `umpan balik` sebelum memulai PR apa pun.

#### Saya bingung, dapatkah Anda membantu saya?

Ya, kamu dapat kirimkan pertanyaanmu di kanal komunitas NusaCrowd! Silakan bergabung ke kanal komunitas NusaCrowd di [grup WhatsApp kami](https://chat.whatsapp.com/Jn4nM6l3kSn3p4kJVESTwv) dan [server Slack](https://join.slack.com/t/nusacrowd/shared_invite/zt-1bbvt4och-JkC7tzeL_eUk4UD6tl3kDg).


## Terima kasih!

Kami sangat menghargai bantuan Anda!

Artefak hackathon ini akan dijelaskan dalam makalah penelitian akademis mendatang yang menargetkan machine learning atau NLP audiens. Silakan merujuk ke [bagian ini](#contribution-guidelines) untuk imbalan kontribusi Anda karena membantu Nusantara NLP. Kami menyadari bahwa beberapa dataset memerlukan lebih banyak upaya daripada yang lain, jadi hubungi kami jika Anda memiliki pertanyaan. Tujuan kami adalah menjadi inklusif dengan kredit!

<!--
## Ucapan Terima Kasih

Panduan hackathon ini sangat terinspirasi oleh [BigScience Datasets Hackathon](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).
 -->
