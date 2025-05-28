import pandas as pd
import numpy as np
from keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import os

class NusaxIndonesiaDataset:
    def __init__(self, token_maksimum=10000, panjang_sekuens=250):
        # Path
        lokasi_sekarang = os.path.abspath(__file__) 
        folder_dataset = os.path.dirname(lokasi_sekarang) 
        data_rnn = os.path.join(folder_dataset, "rnn")
        
        self.file_train = os.path.join(data_rnn, "train.csv")
        self.file_validasi = os.path.join(data_rnn, "valid.csv")
        self.file_test = os.path.join(data_rnn, "test.csv")
        
        # Konfigurasi
        self.token_maksimum = token_maksimum
        self.panjang_sekuens = panjang_sekuens
        
        # Data Variable
        self.x_train_text = []
        self.y_train_text = []
        self.x_val_text = []
        self.y_val_text = []
        self.x_test_text = []
        self.y_test_text = []
        
        self.y_train = np.array([])
        self.y_val = np.array([])
        self.y_test = np.array([])
        self.x_train = np.array([])
        self.x_val = np.array([])
        self.x_test = np.array([])
        
        self.vectorizer = None
        self.label_encoder = None
        self.vocab = []
        
        self.data_loaded_successfully = self.load_preproses_data()
    
    def load_data_csv(self, path):
        try:
            df = pd.read_csv(path)
            texts = df['text'].fillna('').astype(str).tolist()
            labels = df['label'].astype(str).tolist()
            return texts, labels
        except Exception as e:
            print(f"Error saat memuat {path}: {e}")
            return None, None
    
    def load_preproses_data(self):
        print("1. Memuat Data")
        self.x_train_text, self.y_train_text = self.load_data_csv(self.file_train)
        self.x_val_text, self.y_val_text = self.load_data_csv(self.file_validasi)
        self.x_test_text, self.y_test_text = self.load_data_csv(self.file_test)
        
        print(f"Jumlah data train: {len(self.x_train_text)}")
        print(f"Jumlah data validasi: {len(self.x_val_text)}")
        print(f"Jumlah data test: {len(self.x_test_text)}")

        print("\n2. Label Encoding")
        self.label_encoder = LabelEncoder()
        all_labels_for_fitting = self.y_train_text + self.y_val_text + self.y_test_text
        self.label_encoder.fit(list(set(all_labels_for_fitting)))

        self.y_train = self.label_encoder.transform(self.y_train_text)
        self.y_val = self.label_encoder.transform(self.y_val_text)
        self.y_test = self.label_encoder.transform(self.y_test_text)
        
        print("\n3. Vektorisasi")
        self.vectorizer = TextVectorization(max_tokens=self.token_maksimum, output_sequence_length=self.panjang_sekuens)
        self.vectorizer.adapt(self.x_train_text)
        self.vocab = self.vectorizer.get_vocabulary()
        # print(f"Ukuran vocab: {len(self.vocab)}")

        print("\n4. Tokenisasi data teks")
        self.x_train = self.vectorizer(self.x_train_text).numpy()
        self.x_val = self.vectorizer(self.x_val_text).numpy()
        self.x_test = self.vectorizer(self.x_test_text).numpy()
        return True
    
    def get_data(self):        
        return {
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_val': self.x_val,
            'y_val': self.y_val,
            'x_test': self.x_test,
            'y_test': self.y_test,
            'x_train_text': self.x_train_text,
            'y_train_text': self.y_train_text,
            'x_val_text': self.x_val_text,
            'y_val_text': self.y_val_text,
            'x_test_text': self.x_test_text,
            'y_test_text': self.y_test_text,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'vocab': self.vocab
        }
    
    def print_sample_data(self, num_samples=3):
        if not self.data_loaded_successfully:
            print("Data belum berhasil dimuat.")
            return
            
        print("\n=== Contoh Hasil Tokenisasi ===")
        for i in range(min(num_samples, len(self.x_train_text))):
            print(f"\nTeks Asli (Train Set index {i}):\n{self.x_train_text[i]}")
            print(f"Teks Tokenized (Train Set index {i}):\n{self.x_train[i].tolist()}")
            print(f"Label Asli: {self.y_train_text[i]}, Label Encoded: {self.y_train[i]}")

        print(f"\nBentuk x_train: {self.x_train.shape}")
        print(f"Bentuk y_train: {self.y_train.shape}")
        print(f"Bentuk x_val: {self.x_val.shape}")
        print(f"Bentuk y_val: {self.y_val.shape}")
        print(f"Bentuk x_test: {self.x_test.shape}")
        print(f"Bentuk y_test: {self.y_test.shape}")

        print(f"=== Contoh decode tokenized text ===")
        
        token_contoh = self.x_train[0][:20]
        hasil_decode = [self.vocab[idx] for idx in token_contoh if idx != 0]
        
        print(f"Teks asli (20 kata pertama):\n{' '.join(self.x_train_text[0].split()[:20])}")
        print(f"Teks tokenized yang di-decode:\n{' '.join(hasil_decode)}")


if __name__ == "__main__":
    dataset = NusaxIndonesiaDataset()
    dataset.print_sample_data()
        