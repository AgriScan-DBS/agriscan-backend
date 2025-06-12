# AgriScan Backend

Ini adalah backend untuk AgriScan, sebuah aplikasi web yang dirancang untuk mengidentifikasi penyakit tanaman dari gambar. Dibangun dengan Node.js dan framework Hapi, serta menggunakan model TensorFlow.js untuk melakukan prediksi. Backend ini menangani otentikasi pengguna, unggahan gambar, dan inferensi model.

## Fitur

* **Otentikasi Pengguna**: Menyediakan endpoint untuk registrasi pengguna, login, logout, dan reset kata sandi.
* **Endpoint Aman**: Menggunakan JSON Web Tokens (JWT) untuk mengamankan rute yang dilindungi.
* **Prediksi Penyakit Berbasis Gambar**: Menerima gambar daun tanaman dan mengembalikan prediksi penyakit.
* **Prediksi Top-K**: Memberikan 3 prediksi paling mungkin beserta probabilitasnya.
* **Layanan Email**: Mengirim email untuk reset kata sandi kepada pengguna.
* **Integrasi Database**: Terhubung ke database MongoDB Atlas untuk menyimpan informasi pengguna.
* **Unggahan File**: Menangani `multipart/form-data` untuk unggahan gambar.

## Struktur Proyek

```
.
├── config
│   └── database.js
├── controllers
│   ├── AuthController.js
│   └── InferenceController.js
├── middleware
│   ├── auth.js
│   └── upload.js
├── models
│   └── Auth.js
├── services
│   └── ModelService.js
├── tfjs_model
│   └── model.json
├── utils
│   └── emailService.js
├── .gitignore
├── app.js
├── package.json
├── server.js
└── test-predict.js
```

## Memulai

### Prasyarat

* Node.js (v14 atau lebih baru)
* npm
* Akun MongoDB Atlas dan URI koneksi

### Instalasi

1.  **Kloning repositori:**

    ```bash
    git clone https://github.com/AgriScan-DBS/agriscan-backend.git
    cd agriscan-backend
    ```

2.  **Instal dependensi:**

    ```bash
    npm install
    ```

3.  **Siapkan variabel lingkungan:**

    Buat file `.env` di root proyek dan tambahkan variabel berikut:

    ```env
    MONGO_URI=your_mongodb_atlas_connection_uri
    JWT_SECRET=your_jwt_secret
    GMAIL_USER=your_gmail_address
    GMAIL_APP_PASSWORD=your_gmail_app_password
    FRONTEND_URL=your_frontend_application_url
    ```

### Menjalankan Server

Untuk memulai server pengembangan, jalankan:

```bash
npm start
```

Server akan dimulai di `http://localhost:5000` secara default.

## Endpoint API

### Otentikasi

* `POST /api/auth/register`: Mendaftarkan pengguna baru.
* `POST /api/auth/login`: Masuk sebagai pengguna dan menerima JWT.
* `POST /api/auth/logout`: Keluar dari akun pengguna saat ini (memerlukan otentikasi).
* `POST /api/auth/request-reset`: Meminta email untuk reset kata sandi.
* `POST /api/auth/reset-password`: Mereset kata sandi pengguna dengan token yang valid.

### Prediksi

* `POST /api/predict`: Mengunggah gambar untuk prediksi penyakit (memerlukan otentikasi).

## Teknologi yang Digunakan

* **Node.js**: Lingkungan runtime JavaScript.
* **Hapi**: Kerangka kerja yang kaya untuk membangun aplikasi dan layanan.
* **MongoDB**: Database NoSQL untuk menyimpan data pengguna.
* **Mongoose**: Pustaka pemodelan data objek (ODM) untuk MongoDB dan Node.js.
* **TensorFlow.js**: Pustaka untuk machine learning di JavaScript.
* **JSON Web Token (JWT)**: Cara yang ringkas dan aman untuk merepresentasikan klaim yang akan ditransfer antara dua pihak.
* **Bcrypt.js**: Pustaka untuk hashing kata sandi.
* **Nodemailer**: Modul untuk aplikasi Node.js yang memungkinkan pengiriman email dengan mudah.
* **Multer**: Middleware Node.js untuk menangani `multipart/form-data`.

## Model

Aplikasi ini menggunakan model grafik TensorFlow.js yang sudah dilatih sebelumnya untuk klasifikasi gambar. Model ini terletak di direktori `tfjs_model`. `ModelService.js` menangani pemuatan model, pra-pemrosesan gambar input, dan pemrosesan hasil prediksi.

### Pra-pemrosesan Gambar

Sebelum dimasukkan ke dalam model, gambar input akan:

1.  Didekode menjadi tensor 3-channel.
2.  Ukurannya diubah menjadi 224x224 piksel.
3.  Dinormalisasi ke rentang [0, 1].
4.  Dinormalisasi menggunakan nilai mean dan standar deviasi ResNet50.

### Hasil Prediksi

Endpoint prediksi mengembalikan 3 prediksi teratas beserta nama kelas dan probabilitasnya.

## Pengujian

Sebuah skrip pengujian, `test-predict.js`, disediakan untuk menguji endpoint prediksi. Cara menggunakannya:

1.  Pastikan server berjalan.
2.  Letakkan gambar uji bernama `test-image.png` di root proyek.
3.  Perbarui `userName` dan `password` di `test-predict.js` dengan kredensial yang valid.
4.  Jalankan skrip:

    ```bash
    node test-predict.js
    ```

Skrip akan mencatat respons dari endpoint prediksi ke konsol.
