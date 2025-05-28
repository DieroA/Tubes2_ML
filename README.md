# CNN dan RNN

Tugas Besar 2 IF3270 Pembelajaran Mesin

## Deskripsi

Repository ini berisi implementasi CNN, Simple RNN, dan LSTM menggunakan model yang disediakan oleh library Keras. Pelatihan model dilakukan menggunakan dataset berikut:  
| Model | Dataset |  
|------ | ------- |
| CNN | CIFAR-10 |
| Simple RNN | NusaX-Sentiment (Bahasa Indonesia)
| LSTM | NusaX-Sentiment (Bahasa Indonesia) |

Data bobot yang didapatkan melalui pelatihan akan disimpan dan digunakan untuk melakukan Forward Propagation yang diimplementasikan dari awal (_from scratch_).

## Cara setup

- Install Python 3.11.9 (Untuk library TensorFlow)
- Buka terminal dan jalankan perintah berikut

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Cara menjalankan program

- Buka folder src
- Jalankan file .ipynb (e.g. CNN.ipynb)

## Pembagian Tugas

| NIM      | Nama                                | Tugas yang dikerjakan |
| -------- | ----------------------------------- | --------------------- |
| 13522028 | Panji Sri Kuncara Wisma             | Simple RNN            |
| 13522056 | Diero Arga Purnama                  | CNN, File ReadME      |
| 13522120 | Muhammad Rifki Virziadeili Harisman | LSTM                  |
