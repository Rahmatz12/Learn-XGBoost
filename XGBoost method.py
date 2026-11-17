"""XGBoost surrogate untuk memprediksi muka air long storage satu langkah ke depan.

Saat dijalankan, pilih data dummy atau masukkan path CSV data asli.
Kolom CSV wajib: waktu, inflow, hujan, suhu, tinggi_kali_kedurus, q_pompa, muka_air.
Setiap eksekusi menyimpan output terminal ke file baru di folder run_logs.

Jalankan dengan: python "XGBoost method.py"
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

RANDOM_STATE = 42
DATA_PATH = Path("hasil_hecras.csv")
KOLOM_WAJIB = (
    "waktu",
    "inflow",
    "hujan",
    "suhu",
    "tinggi_kali_kedurus",
    "q_pompa",
    "muka_air",
)
TINGGI_BANJIR_ASUMSI_M = 0.40
KALI_KEDURUS_NORMAL_DUMMY_M = 1.20
KALI_KEDURUS_WASPADA_DUMMY_M = 1.50
KALI_KEDURUS_TINGGI_DUMMY_M = 1.80
KAPASITAS_POMPA_M3S = 2.5
POMPA_1_ON_M = 0.40
POMPA_1_OFF_M = 0.30
POMPA_2_ON_M = 0.55
POMPA_2_OFF_M = 0.45
POMPA_ON_WASPADA = (0.45, 0.60)
POMPA_OFF_WASPADA = (0.35, 0.50)
POMPA_ON_TINGGI = (0.50, 0.65)
POMPA_OFF_TINGGI = (0.40, 0.55)
AMBANG_HUJAN_DERAS_MM_JAM = 10.0
SUHU_NORMAL_C = 30.0
SUHU_HUJAN_DERAS_C = 24.0


def ambang_pompa_dummy(tinggi_kali: float) -> tuple[float, float, float, float]:
    """Ambang simulasi; level sungai tinggi menaikkan ambang start pompa."""
    if tinggi_kali >= KALI_KEDURUS_TINGGI_DUMMY_M:
        return (*POMPA_ON_TINGGI, *POMPA_OFF_TINGGI)
    if tinggi_kali >= KALI_KEDURUS_WASPADA_DUMMY_M:
        return (*POMPA_ON_WASPADA, *POMPA_OFF_WASPADA)
    return POMPA_1_ON_M, POMPA_2_ON_M, POMPA_1_OFF_M, POMPA_2_OFF_M


def buat_data_dummy(jumlah: int = 3000) -> pd.DataFrame:
    """Buat data dummy sekitar 40 cm, bukan untuk validasi hidraulik HEC-RAS."""
    rng = np.random.default_rng(RANDOM_STATE)
    waktu = pd.date_range("2024-01-01", periods=jumlah, freq="h")
    hujan = np.clip(rng.gamma(0.3, 6, jumlah) * (rng.random(jumlah) < 0.25), 0, None)
    inflow = (
        np.convolve(hujan, np.ones(6) / 3, mode="same")
        + 1.0
        + rng.normal(0, 0.1, jumlah)
    )
    hujan_kum_6j = pd.Series(hujan).rolling(6, min_periods=1).sum().to_numpy()
    tinggi_kali_kedurus = np.clip(
        KALI_KEDURUS_NORMAL_DUMMY_M
        + 0.025 * hujan_kum_6j
        + rng.normal(0, 0.04, jumlah),
        0.7,
        2.4,
    )
    jam_harian = np.arange(jumlah) % 24
    suhu_normal = (
        SUHU_NORMAL_C
        + 2.0 * np.sin(2 * np.pi * (jam_harian - 14) / 24)
        + rng.normal(0, 0.7, jumlah)
    )
    hujan_deras = hujan >= AMBANG_HUJAN_DERAS_MM_JAM
    suhu = np.where(
        hujan_deras,
        SUHU_HUJAN_DERAS_C + rng.normal(0, 0.5, jumlah),
        suhu_normal,
    )
    q_pompa = np.zeros(jumlah)
    muka_air = np.zeros(jumlah)
    muka_air[0] = TINGGI_BANJIR_ASUMSI_M
    inflow_std = max(float(np.std(inflow)), 1e-6)
    anomali_inflow = np.clip((inflow - np.mean(inflow)) / inflow_std, -3, 3)
    pompa_1_aktif = False
    pompa_2_aktif = False

    for langkah in range(1, jumlah):
        level_sebelumnya = muka_air[langkah - 1]
        ambang_p1_on, ambang_p2_on, ambang_p1_off, ambang_p2_off = ambang_pompa_dummy(
            tinggi_kali_kedurus[langkah]
        )
        if not pompa_1_aktif and level_sebelumnya >= ambang_p1_on:
            pompa_1_aktif = True
        elif pompa_1_aktif and level_sebelumnya <= ambang_p1_off:
            pompa_1_aktif = False

        if not pompa_2_aktif and level_sebelumnya >= ambang_p2_on:
            pompa_2_aktif = True
        elif pompa_2_aktif and level_sebelumnya <= ambang_p2_off:
            pompa_2_aktif = False

        jumlah_pompa_aktif = int(pompa_1_aktif) + int(pompa_2_aktif)
        q_pompa[langkah] = jumlah_pompa_aktif * KAPASITAS_POMPA_M3S

        # Respons muka air ini sintetis; konversi debit ke level perlu luas tampungan.
        muka_air[langkah] = np.clip(
            TINGGI_BANJIR_ASUMSI_M
            + 0.85 * (level_sebelumnya - TINGGI_BANJIR_ASUMSI_M)
            + 0.04 * anomali_inflow[langkah]
            + 0.02
            * (tinggi_kali_kedurus[langkah] - KALI_KEDURUS_NORMAL_DUMMY_M)
            - 0.015 * jumlah_pompa_aktif
            + rng.normal(0, 0.003),
            0.0,
            0.8,
        )

    return pd.DataFrame(
        {
            "waktu": waktu,
            "hujan": hujan,
            "suhu": suhu,
            "tinggi_kali_kedurus": tinggi_kali_kedurus,
            "inflow": inflow,
            "q_pompa": q_pompa,
            "muka_air": muka_air,
        }
    )


def tampilkan_ringkasan_dummy(data_dummy: pd.DataFrame) -> None:
    print(
        f"Asumsi awal muka air: {TINGGI_BANJIR_ASUMSI_M:.2f} m "
        f"({TINGGI_BANJIR_ASUMSI_M * 100:.0f} cm). "
        f"Rentang dummy: {data_dummy['muka_air'].min():.2f}-"
        f"{data_dummy['muka_air'].max():.2f} m."
    )
    jumlah_hujan_deras = int(
        (data_dummy["hujan"] >= AMBANG_HUJAN_DERAS_MM_JAM).sum()
    )
    print(
        f"Asumsi suhu dummy: {SUHU_HUJAN_DERAS_C:.1f} C saat hujan >= "
        f"{AMBANG_HUJAN_DERAS_MM_JAM:.1f} mm/jam; selain itu sekitar "
        f"{SUHU_NORMAL_C:.1f} C. Hujan deras: {jumlah_hujan_deras} jam."
    )
    print(
        "Asumsi tinggi Kali Kedurus dummy: "
        f"normal {KALI_KEDURUS_NORMAL_DUMMY_M:.2f} m, "
        f"waspada >= {KALI_KEDURUS_WASPADA_DUMMY_M:.2f} m, "
        f"tinggi >= {KALI_KEDURUS_TINGGI_DUMMY_M:.2f} m. "
        f"Rentang dummy: {data_dummy['tinggi_kali_kedurus'].min():.2f}-"
        f"{data_dummy['tinggi_kali_kedurus'].max():.2f} m."
    )
    status_kali = np.select(
        [
            data_dummy["tinggi_kali_kedurus"] >= KALI_KEDURUS_TINGGI_DUMMY_M,
            data_dummy["tinggi_kali_kedurus"] >= KALI_KEDURUS_WASPADA_DUMMY_M,
        ],
        ["TINGGI", "WASPADA"],
        default="NORMAL",
    )
    status_gabungan = np.select(
        [
            (data_dummy["muka_air"] >= POMPA_2_ON_M)
            & (data_dummy["tinggi_kali_kedurus"] >= KALI_KEDURUS_TINGGI_DUMMY_M),
            (data_dummy["muka_air"] >= POMPA_2_ON_M)
            | (data_dummy["tinggi_kali_kedurus"] >= KALI_KEDURUS_TINGGI_DUMMY_M),
            (data_dummy["muka_air"] >= POMPA_1_ON_M)
            | (data_dummy["tinggi_kali_kedurus"] >= KALI_KEDURUS_WASPADA_DUMMY_M),
        ],
        ["KRITIS", "TINGGI", "WASPADA"],
        default="NORMAL",
    )
    print("Status Kali Kedurus (dummy):")
    for status, proporsi in pd.Series(status_kali).value_counts(normalize=True).items():
        print(f"  {status}: {proporsi:.1%}")
    print("Risiko gabungan long storage + Kali Kedurus (dummy):")
    for status, proporsi in pd.Series(status_gabungan).value_counts(normalize=True).items():
        print(f"  {status}: {proporsi:.1%}")
    print(
        f"Pola dummy: P1 ON >= {POMPA_1_ON_M:.2f} m/OFF <= {POMPA_1_OFF_M:.2f} m; "
        f"P2 ON >= {POMPA_2_ON_M:.2f} m/OFF <= {POMPA_2_OFF_M:.2f} m; "
        f"kapasitas masing-masing {KAPASITAS_POMPA_M3S:.1f} m3/det."
    )
    print(
        "Penyesuaian ambang dummy saat Kali Kedurus waspada: "
        f"ON P1/P2={POMPA_ON_WASPADA[0]:.2f}/{POMPA_ON_WASPADA[1]:.2f} m; "
        "saat tinggi: "
        f"ON P1/P2={POMPA_ON_TINGGI[0]:.2f}/{POMPA_ON_TINGGI[1]:.2f} m."
    )
    for debit, proporsi in data_dummy["q_pompa"].value_counts(normalize=True).sort_index().items():
        if np.isclose(debit, 0.0):
            mode = "kedua pompa OFF"
        elif np.isclose(debit, KAPASITAS_POMPA_M3S):
            mode = "Pompa 1 ON"
        else:
            mode = "Pompa 1 dan 2 ON"
        print(f"  {mode}: {proporsi:.1%} dari waktu dummy")
    tabel_operasi = pd.crosstab(
        pd.Series(status_kali, name="Status Kali Kedurus"),
        pd.Series(data_dummy["q_pompa"], name="Debit pompa m3/s"),
        normalize="index",
    )
    print("Proporsi operasi pompa per status Kali Kedurus:")
    print((100 * tabel_operasi).round(1).to_string())


def muat_data(path: Path | None = None) -> pd.DataFrame:
    """Pilih data dummy atau baca CSV real berdasarkan input saat program berjalan."""
    if path is None:
        print("\nPilih sumber data:")
        print("1. Data dummy")
        print("2. Data real dari CSV")
        pilihan = input("Masukkan pilihan (1/2, default 1): ").strip() or "1"

        if pilihan == "1":
            data_dummy = buat_data_dummy()
            print("Menggunakan data dummy.")
            tampilkan_ringkasan_dummy(data_dummy)
            return data_dummy
        if pilihan != "2":
            raise ValueError("Pilihan tidak valid. Jalankan ulang dan pilih 1 atau 2.")

        lokasi_csv = input(f"Path CSV [{DATA_PATH}]: ").strip().strip("\"'")
        path = Path(lokasi_csv).expanduser() if lokasi_csv else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"File CSV tidak ditemukan: {path}. Periksa path lalu jalankan ulang."
        )
    data = pd.read_csv(path)
    kolom_hilang = set(KOLOM_WAJIB) - set(data.columns)
    if kolom_hilang:
        raise ValueError(
            "Kolom wajib tidak ditemukan di CSV: " + ", ".join(sorted(kolom_hilang))
        )

    data = data.loc[:, list(KOLOM_WAJIB)].copy()
    data["waktu"] = pd.to_datetime(data["waktu"], errors="coerce")
    for kolom in (
        "inflow",
        "hujan",
        "suhu",
        "tinggi_kali_kedurus",
        "q_pompa",
        "muka_air",
    ):
        data[kolom] = pd.to_numeric(data[kolom], errors="coerce")

    data = data.dropna(subset=list(KOLOM_WAJIB))
    data = data.sort_values("waktu").reset_index(drop=True)
    if len(data) < 100:
        raise ValueError("Data valid terlalu sedikit; dibutuhkan minimal 100 baris.")
    return data


def buat_fitur(
    data: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 6),
    horizon: int = 1,
) -> pd.DataFrame:
    """Buat fitur historis dan target muka air beberapa langkah ke depan."""
    if horizon < 1:
        raise ValueError("horizon harus minimal 1 langkah ke depan.")

    hasil = data.copy()
    for lag in lags:
        hasil[f"muka_air_lag{lag}"] = hasil["muka_air"].shift(lag)
        hasil[f"inflow_lag{lag}"] = hasil["inflow"].shift(lag)

    hasil["tinggi_kali_kedurus_lag1"] = hasil["tinggi_kali_kedurus"].shift(1)
    hasil["tinggi_kali_kedurus_lag3"] = hasil["tinggi_kali_kedurus"].shift(3)
    hasil["hujan_kum_6j"] = hasil["hujan"].rolling(6).sum()
    hasil["hujan_kum_24j"] = hasil["hujan"].rolling(24).sum()
    hasil["target"] = hasil["muka_air"].shift(-horizon)
    return hasil.dropna().reset_index(drop=True)


def hitung_nse(observasi: np.ndarray, prediksi: np.ndarray) -> float:
    penyebut = np.sum((observasi - observasi.mean()) ** 2)
    if penyebut == 0:
        return float("nan")
    return float(1 - np.sum((observasi - prediksi) ** 2) / penyebut)


def evaluasi(nama: str, observasi: pd.Series, prediksi: np.ndarray) -> None:
    rmse = np.sqrt(mean_squared_error(observasi, prediksi))
    mae = mean_absolute_error(observasi, prediksi)
    r2 = r2_score(observasi, prediksi)
    nse = hitung_nse(observasi.to_numpy(), prediksi)
    print(f"{nama:5s} | RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  NSE={nse:.4f}")


def jalankan_pipeline() -> None:
    data = buat_fitur(muat_data())
    fitur = [kolom for kolom in data.columns if kolom not in ("waktu", "target")]
    X, y = data[fitur], data["target"]

    # Pembagian berurutan menjaga data masa depan tidak masuk ke data latih.
    jumlah_latih = int(len(data) * 0.70)
    jumlah_validasi = int(len(data) * 0.15)
    batas_test = jumlah_latih + jumlah_validasi
    if jumlah_latih == 0 or jumlah_validasi == 0 or batas_test >= len(data):
        raise ValueError("Data tidak cukup untuk pembagian train/validation/test.")

    X_train, y_train = X.iloc[:jumlah_latih], y.iloc[:jumlah_latih]
    X_validasi = X.iloc[jumlah_latih:batas_test]
    y_validasi = y.iloc[jumlah_latih:batas_test]
    X_test, y_test = X.iloc[batas_test:], y.iloc[batas_test:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validasi, y_validasi)],
        verbose=False,
    )
    print(f"Jumlah pohon terpakai: {model.best_iteration + 1}")

    print("\n=== Hasil model dasar ===")
    evaluasi("Train", y_train, model.predict(X_train))
    evaluasi("Val", y_validasi, model.predict(X_validasi))
    evaluasi("Test", y_test, model.predict(X_test))

    distribusi_parameter = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [0.5, 1, 5],
    }
    pencarian = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        param_distributions=distribusi_parameter,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    X_train_validasi = pd.concat([X_train, X_validasi])
    y_train_validasi = pd.concat([y_train, y_validasi])
    pencarian.fit(X_train_validasi, y_train_validasi)
    print("\nParameter terbaik:", pencarian.best_params_)

    model_terbaik = pencarian.best_estimator_
    prediksi_test = model_terbaik.predict(X_test)
    print("\n=== Hasil tuning pada data test ===")
    evaluasi("Test", y_test, prediksi_test)

    pentingnya_fitur = pd.Series(
        model_terbaik.feature_importances_, index=fitur
    ).sort_values()
    pentingnya_fitur.plot(
        kind="barh", figsize=(7, 5), title="Feature Importance XGBoost"
    )
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    waktu_test = data.loc[X_test.index, "waktu"]
    jumlah_plot = min(300, len(y_test))
    plt.figure(figsize=(10, 4))
    plt.plot(waktu_test.iloc[:jumlah_plot], y_test.iloc[:jumlah_plot], label="HEC-RAS")
    plt.plot(
        waktu_test.iloc[:jumlah_plot],
        prediksi_test[:jumlah_plot],
        "--",
        label="XGBoost",
    )
    plt.xlabel("Waktu")
    plt.ylabel("Muka air (m)")
    plt.legend()
    plt.title("Observasi vs prediksi muka air")
    plt.tight_layout()
    plt.savefig("obs_vs_pred.png", dpi=150)
    plt.close()

    model_terbaik.save_model("xgb_surrogate.json")
    print("\nModel disimpan: xgb_surrogate.json")

    contoh = model_terbaik.predict(X_test.iloc[[0]])[0]
    print(f"Contoh prediksi muka air t+1: {contoh:.4f} m")


class PencatatGanda:
    def __init__(self, terminal, file_log) -> None:
        self.terminal = terminal
        self.file_log = file_log

    def write(self, teks: str) -> int:
        self.terminal.write(teks)
        self.file_log.write(teks)
        return len(teks)

    def flush(self) -> None:
        self.terminal.flush()
        self.file_log.flush()

    def __getattr__(self, nama: str):
        return getattr(self.terminal, nama)


def main() -> None:
    folder_log = Path(__file__).resolve().parent / "run_logs"
    folder_log.mkdir(exist_ok=True)
    waktu_run = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path_log = folder_log / f"run_{waktu_run}.txt"

    with path_log.open("w", encoding="utf-8") as file_log:
        output = PencatatGanda(sys.stdout, file_log)
        error = PencatatGanda(sys.stderr, file_log)
        with redirect_stdout(output), redirect_stderr(error):
            print(f"Log run ini: {path_log}")
            try:
                jalankan_pipeline()
            except Exception:
                traceback.print_exc()
                raise
            finally:
                print(f"\nOutput run tersimpan di: {path_log}")


if __name__ == "__main__":
    main()