# Mengunggah skrip data loader ke Hub

**Pada titik ini, seharusnya tidak ada perubahan lebih lanjut pada skrip data loader Anda setelah PR diterima**.

### 1. Buat akun di 🤗's Hub

Harap lakukan hal berikut sebelum memulai:

- [Buat](https://huggingface.co/join) akun di 🤗's Hub dan [login](https://huggingface.co/login). **Pilih kata sandi yang baik, karena Anda perlu mengautentikasi kredensial Anda**.

- Bergabunglah dengan Indobenchmark [di sini](https://huggingface.co/indobenchmark).
    - klik tombol "Request to join this org" di sudut kanan atas.

- Buat akun github; Anda dapat mengikuti petunjuk untuk menginstal git [di sini](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).


**Catatan - izin Anda akan disetel ke READ. Silakan hubungi admin di masalah github dataset Anda untuk diberikan akses WRITE; ini harus diberikan setelah PR Anda diterima**.

### 2) Aktifkan Huggingface hub

Anda dapat menemukan petunjuk resmi [di sini](https://huggingface.co/welcome). Kami akan menyediakan apa yang Anda butuhkan untuk lingkungan hackathon nusantara-datasets.

Aktifkan environment `nusantara` Anda, gunakan perintah berikut:

```
huggingface-cli login
```

Masuk dengan nama pengguna dan kata sandi akun 🤗 Hub Anda.

### 3. Buat repositori dataset

Buat repositori melalui Hub [di sini](https://huggingface.co/new-dataset) dengan detail berikut.

+ Set Owner: nusantara-datasets
+ Set Dataset name: the name of the dataset
+ Set License: the license that applies to this dataset
+ Select Private
+ Click `Create dataset`

**Harap beri nama skrip data loader Anda dengan nama yang sama dengan dataset.** Misalnya, jika skrip pemuat dataset Anda disebut `absa_prosa.py`, maka nama dataset Anda harus `absa_prosa`.

Jika tidak ada lisensi yang sesuai yang tersedia dalam opsi yang disediakan (misalnya untuk dataset dengan perjanjian pengguna data tertentu), Anda harus memilih "other".

### 4. Clone repositori dataset

Menggunakan akses terminal, temukan lokasi untuk menempatkan repositori github Anda. Di lokasi ini, gunakan perintah berikut:

```
git clone https://huggingface.co/datasets/indobenchmark/<nama_dataset_anda>
```

### 5. Lakukan perubahan Anda

Jalankan perintah berikut untuk menambah dan mendorong pekerjaan Anda.

```
git add <nama_file_anda.py> # tambahkan dataset
git commit -m "Menambahkan <nama_dataset_anda>"
git push origin
```

## 6) Uji data loader Anda

Jalankan perintah berikut **dalam folder yang tidak menyertakan skrip data loader**:

Uji dataset baik itu dengan skema/config asli maupun skema/config nusantara.

**Dataset Publik**
```python
from datasets import load_dataset

dataset_orig = load_dataset("indobenchmark/<nama_dataset_anda>", name="source", use_auth_token=True)
dataset_indobenchmark= load_dataset("indobenchmark/<nama_dataset_anda>", name="indobenchmark", use_auth_token=True)
```

**Set Data Pribadi**

```python
from datasets import load_dataset

dataset_orig = load_dataset(
    "indobenchmark/<nama_dataset_anda>",
    name="source",
    data_dir="/local/path/menuju/data/files",
    use_auth_token=True)

dataset_indobenchmark = load_dataset(
    "indobenchmark/<nama_dataset_anda>",
    name="indobenchmark",
    data_dir="/local/path/menuju/data/files",
    use_auth_token=True)
```

Dan dengan ini, Anda telah berhasil berkontribusi dengan membuat data-loading script!
