"""
EOL8709R, EOL8709Q ve EOL8709U hücreleriyle Random Forest ölçüm modeli eğitimi,
R-Q-U hücrelerinin ortadaki uzun ve sürekli ortak rotası üzerinde CTRV Particle Filter takibi.

Bu kodda:
1) Azimuth, Random Forest çıktısında kullanılmaz.
2) Ham GPS koordinatları temizlenir ve 1 saniyelik hücresel ölçüm zamanlarına
   doğrusal olarak enterpole edilir. Uzun zaman boşlukları ayrı segmentlerdir.
3) R, Q ve U hücrelerinin her biri ayrı ayrı %70/%30, %80/%20 ve %90/%10
   oranlarında bölünür; en iyi split ve Random Forest adayı seçilir.
4) Görseldeki sağ taraftaki kopuk noktalar otomatik olarak dışarıda bırakılır;
   R-Q-U hücrelerinin ortadaki uzun ve sürekli rotası takip edilir.
5) Random Forest girişi yalnız [x_m, y_m], çıkışı 8 radyo parametresidir.
6) Çıktı olarak yalnızca split tablosu, takip grafiği ve hata CDF grafiği gösterilir.

RF girişi  : [x_m, y_m]
RF çıkışı  : [RSRP, RSRQ, SINR, CQI, Timing Advance,
              MCS0, MCS1, PUSCH Tx Power]
PF ölçümü  : Aynı 8 hücresel parametre.
"""

import os
import math
import re
import zipfile
import warnings
from pathlib import Path, PurePosixPath

from lxml import etree as LET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.covariance import LedoitWolf
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")


EXCEL_PATH = Path("ISTANBUL_DATA-MARMARA.xlsx")

SHEET_MEASUREMENTS = "Sheet1"
SHEET_ANTENNAS = "Sheet2"

TRAIN_CELLS = ["EOL8709R", "EOL8709Q", "EOL8709U"]
TRACK_CELLS = TRAIN_CELLS.copy()
TRACK_LABEL = "EOL8709R-Q-U merkezi uzun rota"
ALL_CELLS = TRAIN_CELLS.copy()

# Görselde sağ tarafta kalan kopuk noktaları dışarıda bırakmak için ön kapı.
# Merkezdeki uzun rota yaklaşık x <= 500 m bölgesindedir.
TRACK_X_MAX_M = 500.0

# Kalan noktalar içinden ortadaki uzun ve bağlı rotayı seçmek için uzamsal ayarlar.
TRACK_DBSCAN_EPS_M = 45.0
TRACK_DBSCAN_MIN_SAMPLES = 3
TRACK_MAX_TIME_GAP_S = 5.0
TRACK_MAX_SPATIAL_JUMP_M = 80.0

RESAMPLE_MS = 1000

# GPS zaman düzeltmesi
# Ham hücresel ölçümler sık, GPS koordinatları ise daha seyrek güncelleniyor.
# GPS konumları temizlendikten sonra 1 saniyelik zaman ızgarasına enterpole edilir.
MAX_CONTINUOUS_GAP_S = 2.0       # Daha büyük zaman boşlukları ayrı segmenttir
GPS_CHANGE_THRESHOLD_M = 0.05    # Aynı GPS koordinatının tekrarlarını ayırma eşiği
GPS_SPIKE_MIN_DISTANCE_M = 25.0  # Tek-adımlık sıçrama adayının asgari mesafesi
GPS_SPIKE_RETURN_RATIO = 0.35    # Önceki-sonraki direkt yol / sıçrama yolu oranı
GPS_MAX_PLAUSIBLE_SPEED_MPS = 60.0

TRAIN_RATIOS = [0.70, 0.80, 0.90]
RANDOM_STATE = 42

# Hiperparametre araması: 3 split için makul süre / güçlü overfitting kontrolü
N_JOBS = max(1, min(8, os.cpu_count() or 1))

# Particle Filter
N_PARTICLES = 1000
RF_GRID_STEP_M = 5.0             # PF içinde hızlı RF ölçüm haritası çözünürlüğü
RESAMPLE_THRESHOLD = 0.50       # N_eff < 0.5*N ise resampling
LIKELIHOOD_TEMPERATURE = 3.0    # Ölçüm/model uyuşmazlığında ağırlık çökmesini yumuşatır
STUDENT_T_DOF = 5.0             # Ağır kuyruklu ölçüm olasılığı
R_INFLATION = 1.0               # U eğitimde görüldüğü için özel kovaryans büyütmesi yok

# İlk gerçek takip noktası PF başlangıç merkezi olarak kullanılır.
# False olduğunda ileri zamandaki gerçek konumlardan hız/yön alınmaz.
USE_FIRST_MOTION_FOR_INITIALIZATION = False
INITIAL_POSITION_STD_M = 15.0
INITIAL_SPEED_MEAN_MPS = 8.0
INITIAL_SPEED_STD_MPS = 4.0
INITIAL_HEADING_STD_DEG = 30.0
INITIAL_TURN_RATE_STD_DEG_S = 8.0

# CTRV süreç gürültüsü standart sapmaları (1 saniye için; dt ile sqrt(dt) ölçeklenir)
PROCESS_STD_1S = np.array([
    0.80,                   # x [m]
    0.80,                   # y [m]
    1.20,                   # v [m/s]
    np.deg2rad(5.0),        # theta [rad]
    np.deg2rad(3.0),        # omega [rad/s]
], dtype=float)

# Resampling sonrası parçacık fakirleşmesini azaltan küçük roughening
ROUGHENING_STD = np.array([
    0.35,
    0.35,
    0.20,
    np.deg2rad(1.0),
    np.deg2rad(0.5),
], dtype=float)

EARTH_RADIUS_M = 6_371_000.0

TIME_COL = "TIME"
LAT_COL = "LAT"
LON_COL = "LON"
CELL_COL = "NR_CELL_NAME_ENCODED"

RADIO_OUTPUTS = [
    "NR_RSRP_0",
    "NR_RSRQ_0",
    "NR_SINR_0",
    "CQI_NR",
    "NR_TIMING_ADVANCE",
    "NR_PDSCH_MCS_0",
    "NR_PDSCH_MCS_1",
    "PuschTxPower",
]

INTERNAL_OUTPUTS = RADIO_OUTPUTS.copy()
RF_INPUTS = ["x_m", "y_m"]

# Ölçüm kovaryansı için asgari standart sapmalar.
# Holdout artıkları yapay olarak çok küçük çıkarsa filtre aşırı güvenmesin.
MEASUREMENT_STD_FLOOR = np.array([
    2.0,   # RSRP [dB]
    0.8,   # RSRQ [dB]
    2.0,   # SINR [dB]
    1.0,   # CQI
    1.0,   # Timing Advance
    1.5,   # MCS0
    1.5,   # MCS1
    1.5,   # PUSCH Tx Power [dB]
], dtype=float)


# =============================================================================
# 2) YARDIMCI FONKSİYONLAR
# =============================================================================

def wrap_angle_rad(angle):
    """Açıyı [-pi, pi) aralığına sarar."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_deg(angle):
    """Açıyı [0, 360) aralığına sarar."""
    return np.mod(angle, 360.0)


def circular_error_deg(pred_deg, true_deg):
    """En kısa yönlü açı hatası: [-180, 180)."""
    return (np.asarray(pred_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0


def normalize_coordinate(value, limit):
    """
    Sheet2'de 40.98250 yerine 4098250 gibi saklanan koordinatları düzeltir.
    limit: enlem için 90, boylam için 180.
    """
    if pd.isna(value):
        return np.nan
    x = float(value)
    while abs(x) > limit:
        x /= 10.0
    return x


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Küçük çalışma alanları için yerel teğet düzlem/equirectangular dönüşümü.
    Baz istasyonu (lat0, lon0) tam olarak (0,0) olur.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0_rad = np.deg2rad(lat0)
    x = EARTH_RADIUS_M * np.cos(lat0_rad) * np.deg2rad(lon - lon0)
    y = EARTH_RADIUS_M * np.deg2rad(lat - lat0)
    return x, y


def local_xy_to_latlon(x, y, lat0, lon0):
    """Yerel x-y koordinatlarını tekrar enlem-boylama çevirir."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lat = lat0 + np.rad2deg(y / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(
        x / (EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)))
    )
    return lat, lon


def extract_gps_control_points(segment_df):
    """
    Bir zaman segmentindeki gerçek GPS güncelleme noktalarını çıkarır.

    Ham dosyada aynı koordinat çok sayıda hücresel satır boyunca tekrar eder.
    Yalnızca koordinat gerçekten değiştiğinde yeni bir kontrol noktası tutulur.
    """
    gps = (
        segment_df[[TIME_COL, "x_m", "y_m"]]
        .dropna()
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    if gps.empty:
        return gps

    # Ardışık aynı konum tekrarlarını kaldır.
    step = np.sqrt(gps["x_m"].diff() ** 2 + gps["y_m"].diff() ** 2)
    keep = step.isna() | (step > GPS_CHANGE_THRESHOLD_M)
    gps = gps.loc[keep].reset_index(drop=True)

    # Son ham örnek, son GPS fix'inin segment sonuna kadar geçerli olduğunu
    # belirtmek için ayrıca tutulur.
    last = segment_df[[TIME_COL, "x_m", "y_m"]].dropna().iloc[-1]
    if gps.empty or last[TIME_COL] > gps[TIME_COL].iloc[-1]:
        gps = pd.concat([gps, last.to_frame().T], ignore_index=True)

    return gps


def remove_single_point_gps_spikes(gps_df):
    """
    Önceki ve sonraki noktaya uzak, fakat önceki-sonraki noktalar birbirine
    yakınsa ortadaki tek-adımlık GPS sıçramasını iteratif olarak kaldırır.
    """
    gps = gps_df.copy().reset_index(drop=True)
    if len(gps) < 3:
        return gps, 0

    removed = 0
    changed = True
    while changed and len(gps) >= 3:
        changed = False
        remove_index = None

        for i in range(1, len(gps) - 1):
            p_prev = gps.loc[i - 1, ["x_m", "y_m"]].to_numpy(dtype=float)
            p_curr = gps.loc[i, ["x_m", "y_m"]].to_numpy(dtype=float)
            p_next = gps.loc[i + 1, ["x_m", "y_m"]].to_numpy(dtype=float)

            d_prev = float(np.linalg.norm(p_curr - p_prev))
            d_next = float(np.linalg.norm(p_next - p_curr))
            d_direct = float(np.linalg.norm(p_next - p_prev))

            dt_prev = max(
                (gps.loc[i, TIME_COL] - gps.loc[i - 1, TIME_COL]).total_seconds(),
                1e-3,
            )
            dt_next = max(
                (gps.loc[i + 1, TIME_COL] - gps.loc[i, TIME_COL]).total_seconds(),
                1e-3,
            )
            speed_prev = d_prev / dt_prev
            speed_next = d_next / dt_next

            is_large_out_and_back = (
                d_prev >= GPS_SPIKE_MIN_DISTANCE_M
                and d_next >= GPS_SPIKE_MIN_DISTANCE_M
                and d_direct <= GPS_SPIKE_RETURN_RATIO * (d_prev + d_next)
            )
            is_impossible_fast_spike = (
                speed_prev > GPS_MAX_PLAUSIBLE_SPEED_MPS
                and speed_next > GPS_MAX_PLAUSIBLE_SPEED_MPS
                and d_direct < max(GPS_SPIKE_MIN_DISTANCE_M, 0.5 * min(d_prev, d_next))
            )

            if is_large_out_and_back or is_impossible_fast_spike:
                remove_index = i
                break

        if remove_index is not None:
            gps = gps.drop(index=remove_index).reset_index(drop=True)
            removed += 1
            changed = True

    return gps, removed


def interpolate_gps_to_times(segment_df, target_times, base_lat, base_lon):
    """Temiz GPS kontrol noktalarını hedef zamanlara doğrusal enterpole eder."""
    gps = extract_gps_control_points(segment_df)
    gps, removed_spikes = remove_single_point_gps_spikes(gps)

    if gps.empty:
        raise ValueError("GPS enterpolasyonu için geçerli koordinat bulunamadı.")

    target_ns = pd.DatetimeIndex(target_times).view("int64").astype(float)
    gps_ns = pd.DatetimeIndex(gps[TIME_COL]).view("int64").astype(float)

    # Aynı zaman damgasında birden fazla koordinat kalırsa kronolojik olarak
    # sonuncuyu seç. Sıçrama temizliği bunu büyük ölçüde zaten çözmüş olur.
    unique_times, unique_indices = np.unique(gps_ns, return_index=True)
    gps = gps.iloc[unique_indices].reset_index(drop=True)
    gps_ns = unique_times

    if len(gps) == 1:
        x_interp = np.full(len(target_ns), float(gps["x_m"].iloc[0]))
        y_interp = np.full(len(target_ns), float(gps["y_m"].iloc[0]))
    else:
        x_interp = np.interp(target_ns, gps_ns, gps["x_m"].to_numpy(dtype=float))
        y_interp = np.interp(target_ns, gps_ns, gps["y_m"].to_numpy(dtype=float))

    lat_interp, lon_interp = local_xy_to_latlon(
        x_interp, y_interp, base_lat, base_lon
    )
    return x_interp, y_interp, lat_interp, lon_interp, len(gps), removed_spikes


def robust_range(values):
    """NRMSE paydası için %5-%95 aralığı; gerekirse std ve 1.0 fallback."""
    values = np.asarray(values, dtype=float)
    q05 = np.nanpercentile(values, 5, axis=0)
    q95 = np.nanpercentile(values, 95, axis=0)
    scale = q95 - q05
    std = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-9, scale, std)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return scale


def systematic_resample(weights, rng):
    """Systematic resampling indeksleri."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    return np.searchsorted(cumulative_sum, positions)


def effective_sample_size(weights):
    return 1.0 / np.sum(np.square(weights))


def weighted_state_mean(particles, weights):
    """Theta için dairesel ortalama kullanan ağırlıklı durum tahmini."""
    estimate = np.zeros(5, dtype=float)
    estimate[0] = np.sum(weights * particles[:, 0])
    estimate[1] = np.sum(weights * particles[:, 1])
    estimate[2] = np.sum(weights * particles[:, 2])
    estimate[3] = np.arctan2(
        np.sum(weights * np.sin(particles[:, 3])),
        np.sum(weights * np.cos(particles[:, 3])),
    )
    estimate[4] = np.sum(weights * particles[:, 4])
    return estimate


def state_covariance(particles, weights, mean_state):
    """Theta farkını sararak ağırlıklı durum kovaryansı."""
    diffs = particles - mean_state
    diffs[:, 3] = wrap_angle_rad(diffs[:, 3])
    return (diffs * weights[:, None]).T @ diffs


def ctrv_predict(particles, dt, rng, xy_bounds):
    """Tüm parçacıkları CTRV modeli ile bir adım ileri taşır."""
    dt = float(np.clip(dt, 0.02, 1.00))

    x = particles[:, 0]
    y = particles[:, 1]
    v = particles[:, 2]
    theta = particles[:, 3]
    omega = particles[:, 4]

    turning = np.abs(omega) > 1e-4
    straight = ~turning

    if np.any(turning):
        om = omega[turning]
        th = theta[turning]
        vv = v[turning]
        x[turning] += (vv / om) * (np.sin(th + om * dt) - np.sin(th))
        y[turning] += (vv / om) * (-np.cos(th + om * dt) + np.cos(th))

    if np.any(straight):
        th = theta[straight]
        vv = v[straight]
        x[straight] += vv * np.cos(th) * dt
        y[straight] += vv * np.sin(th) * dt

    theta[:] = wrap_angle_rad(theta + omega * dt)

    # Süreç gürültüsü
    noise_std = PROCESS_STD_1S * np.sqrt(dt)
    particles += rng.normal(0.0, noise_std, size=particles.shape)
    particles[:, 3] = wrap_angle_rad(particles[:, 3])

    # Fiziksel sınırlar
    particles[:, 2] = np.clip(particles[:, 2], 0.0, 45.0)
    particles[:, 4] = np.clip(particles[:, 4], -1.2, 1.2)
    particles[:, 0] = np.clip(particles[:, 0], xy_bounds[0], xy_bounds[1])
    particles[:, 1] = np.clip(particles[:, 1], xy_bounds[2], xy_bounds[3])


def student_t_log_likelihood(residuals, r_inv, logdet_r, dof):
    """
    Çok değişkenli Student-t log-likelihood'ın parçacıklar arasında değişen kısmı.
    Gaussian'a göre model uyuşmazlığı ve outlier'lara daha dayanıklıdır.
    """
    d = residuals.shape[1]
    mahal = np.einsum("ni,ij,nj->n", residuals, r_inv, residuals, optimize=True)
    mahal = np.maximum(mahal, 0.0)
    return -0.5 * logdet_r - 0.5 * (dof + d) * np.log1p(mahal / dof)


def estimate_initial_motion(track_df):
    """İsteğe bağlı olarak ilk belirgin GPS yer değiştirmesinden v ve theta hesaplar."""
    p0 = track_df[["x_m", "y_m"]].iloc[0].to_numpy(dtype=float)
    t0 = track_df[TIME_COL].iloc[0]

    for j in range(1, min(len(track_df), 50)):
        pj = track_df[["x_m", "y_m"]].iloc[j].to_numpy(dtype=float)
        dist = np.linalg.norm(pj - p0)
        dt = (track_df[TIME_COL].iloc[j] - t0).total_seconds()
        if dist >= 3.0 and dt > 0.05:
            speed = np.clip(dist / dt, 0.0, 35.0)
            heading = math.atan2(pj[1] - p0[1], pj[0] - p0[0])
            return speed, heading

    return INITIAL_SPEED_MEAN_MPS, 0.0


def create_initial_particles(track_df, rng):
    """Yalnız uzun rotanın ilk konumu çevresinde parçacık bulutu oluşturur."""
    x0 = float(track_df["x_m"].iloc[0])
    y0 = float(track_df["y_m"].iloc[0])

    particles = np.zeros((N_PARTICLES, 5), dtype=float)
    particles[:, 0] = rng.normal(x0, INITIAL_POSITION_STD_M, N_PARTICLES)
    particles[:, 1] = rng.normal(y0, INITIAL_POSITION_STD_M, N_PARTICLES)

    if USE_FIRST_MOTION_FOR_INITIALIZATION:
        v0, theta0 = estimate_initial_motion(track_df)
        particles[:, 2] = np.clip(
            rng.normal(v0, INITIAL_SPEED_STD_MPS, N_PARTICLES), 0.0, 45.0
        )
        particles[:, 3] = wrap_angle_rad(
            rng.normal(theta0, np.deg2rad(INITIAL_HEADING_STD_DEG), N_PARTICLES)
        )
    else:
        particles[:, 2] = np.clip(
            rng.normal(INITIAL_SPEED_MEAN_MPS, INITIAL_SPEED_STD_MPS, N_PARTICLES),
            0.0,
            45.0,
        )
        particles[:, 3] = rng.uniform(-np.pi, np.pi, N_PARTICLES)

    particles[:, 4] = rng.normal(
        0.0, np.deg2rad(INITIAL_TURN_RATE_STD_DEG_S), N_PARTICLES
    )
    particles[:, 4] = np.clip(particles[:, 4], -1.2, 1.2)
    return particles


# =============================================================================
# 3) EXCEL OKUMA, GPS TEMİZLEME/ENTERPOLASYON VE 1 s RESAMPLE
# =============================================================================

XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XLSX_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def excel_column_index(cell_reference):
    """A1 hücre referansındaki sütunu sıfır tabanlı indekse çevirir."""
    index = 0
    for char in str(cell_reference):
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - 64)
    return index - 1


def xlsx_sheet_xml_path(zip_file, sheet_name):
    """Sheet adını gerçek xl/worksheets/*.xml yoluna eşler."""
    workbook_root = LET.fromstring(zip_file.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook_root.findall(f".//{{{XLSX_MAIN_NS}}}sheet"):
        if sheet.get("name") == sheet_name:
            relationship_id = sheet.get(f"{{{XLSX_DOC_REL_NS}}}id")
            break
    if relationship_id is None:
        raise KeyError(f"Excel içinde '{sheet_name}' sayfası bulunamadı.")

    rels_root = LET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    target = None
    for rel in rels_root.findall(f"{{{XLSX_PKG_REL_NS}}}Relationship"):
        if rel.get("Id") == relationship_id:
            target = rel.get("Target")
            break
    if target is None:
        raise KeyError(f"'{sheet_name}' sayfasının XML ilişkisi bulunamadı.")

    target_path = PurePosixPath(target)
    if str(target_path).startswith("/"):
        return str(target_path).lstrip("/")
    return str(PurePosixPath("xl") / target_path)


def load_xlsx_shared_strings(zip_file):
    """Shared string tablosunu streaming olarak okur."""
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    strings = []
    si_tag = f"{{{XLSX_MAIN_NS}}}si"
    text_tag = f"{{{XLSX_MAIN_NS}}}t"
    source = zip_file.open("xl/sharedStrings.xml")
    for _, element in LET.iterparse(
        source, events=("end",), tag=si_tag, huge_tree=True
    ):
        strings.append("".join(node.text or "" for node in element.iter(text_tag)))
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    source.close()
    return strings


def fast_read_xlsx_sheet(
    excel_path,
    sheet_name,
    usecols,
    filter_column=None,
    allowed_values=None,
):
    """
    Büyük XLSX dosyasını openpyxl ile tamamen yüklemek yerine XML'i streaming okur.
    Bu dosyada Sheet1 yaklaşık 246 MB XML olduğu için süreyi ciddi biçimde azaltır.
    """
    records = []
    allowed_values = set(allowed_values or [])

    with zipfile.ZipFile(excel_path, "r") as zip_file:
        sheet_xml = xlsx_sheet_xml_path(zip_file, sheet_name)
        shared_strings = load_xlsx_shared_strings(zip_file)

        row_tag = f"{{{XLSX_MAIN_NS}}}row"
        cell_tag = f"{{{XLSX_MAIN_NS}}}c"
        value_tag = f"{{{XLSX_MAIN_NS}}}v"
        text_tag = f"{{{XLSX_MAIN_NS}}}t"

        requested_index_to_name = None
        filter_index = None
        source = zip_file.open(sheet_xml)

        for _, row_element in LET.iterparse(
            source, events=("end",), tag=row_tag, huge_tree=True
        ):
            row_values = {}
            for cell in row_element.findall(cell_tag):
                column_index = excel_column_index(cell.get("r", ""))
                cell_type = cell.get("t")
                value_node = cell.find(value_tag)

                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter(text_tag)
                    )
                elif value_node is None:
                    value = None
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    value = value_node.text

                row_values[column_index] = value

            if requested_index_to_name is None:
                header_name_to_index = {
                    str(value).strip(): index
                    for index, value in row_values.items()
                    if value is not None
                }
                missing = [col for col in usecols if col not in header_name_to_index]
                if missing:
                    raise KeyError(
                        f"{sheet_name} sayfasında bulunamayan sütunlar: {missing}"
                    )
                requested_index_to_name = {
                    header_name_to_index[col]: col for col in usecols
                }
                if filter_column is not None:
                    if filter_column not in header_name_to_index:
                        raise KeyError(
                            f"{sheet_name} sayfasında filtre sütunu yok: {filter_column}"
                        )
                    filter_index = header_name_to_index[filter_column]
            else:
                if (
                    filter_column is None
                    or row_values.get(filter_index) in allowed_values
                ):
                    records.append({
                        name: row_values.get(index)
                        for index, name in requested_index_to_name.items()
                    })

            row_element.clear()
            while row_element.getprevious() is not None:
                del row_element.getparent()[0]

        source.close()

    return pd.DataFrame.from_records(records, columns=usecols)


def load_and_prepare_data(excel_path):
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel dosyası bulunamadı: {excel_path.resolve()}\n"
            "Kod ile Excel dosyasını aynı klasöre koy veya EXCEL_PATH'i değiştir."
        )


    measurement_usecols = [
        TIME_COL,
        LAT_COL,
        LON_COL,
        *RADIO_OUTPUTS,
        CELL_COL,
        "CELL LAT",
        "CELL LON",
    ]
    antenna_usecols = [
        "NR_DU_CELL_NAME_ENCODED",
        "TEKNOLOJI",
        "LATITUDE_DEC",
        "LONGITUDE_DEC",
    ]

    raw = fast_read_xlsx_sheet(
        excel_path=excel_path,
        sheet_name=SHEET_MEASUREMENTS,
        usecols=measurement_usecols,
        filter_column=CELL_COL,
        allowed_values=ALL_CELLS,
    )
    antenna = fast_read_xlsx_sheet(
        excel_path=excel_path,
        sheet_name=SHEET_ANTENNAS,
        usecols=antenna_usecols,
        filter_column="NR_DU_CELL_NAME_ENCODED",
        allowed_values=ALL_CELLS,
    )

    raw.columns = raw.columns.astype(str).str.strip()
    antenna.columns = antenna.columns.astype(str).str.strip()
    raw[CELL_COL] = raw[CELL_COL].astype(str).str.strip()
    antenna["NR_DU_CELL_NAME_ENCODED"] = (
        antenna["NR_DU_CELL_NAME_ENCODED"].astype(str).str.strip()
    )

    if raw.empty:
        raise ValueError(f"Sheet1 içinde {ALL_CELLS} hücreleri bulunamadı.")

    missing_antenna = sorted(
        set(ALL_CELLS) - set(antenna["NR_DU_CELL_NAME_ENCODED"].unique())
    )
    if missing_antenna:
        raise ValueError(f"Sheet2 anten tablosunda bulunamayan hücreler: {missing_antenna}")

    antenna["BS_LAT"] = antenna["LATITUDE_DEC"].apply(
        lambda v: normalize_coordinate(v, 90.0)
    )
    antenna["BS_LON"] = antenna["LONGITUDE_DEC"].apply(
        lambda v: normalize_coordinate(v, 180.0)
    )

    antenna_map = (
        antenna.groupby("NR_DU_CELL_NAME_ENCODED", as_index=True)
        .agg({"BS_LAT": "median", "BS_LON": "median"})
    )

    base_lat = float(antenna_map.loc[ALL_CELLS, "BS_LAT"].median())
    base_lon = float(antenna_map.loc[ALL_CELLS, "BS_LON"].median())

    if not (-90.0 <= base_lat <= 90.0 and -180.0 <= base_lon <= 180.0):
        raise ValueError("Sheet2 baz istasyonu koordinatları geçersiz görünüyor.")

    raw[TIME_COL] = pd.to_datetime(raw[TIME_COL], errors="coerce")
    numeric_cols = [LAT_COL, LON_COL, *RADIO_OUTPUTS]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=[TIME_COL, LAT_COL, LON_COL, *RADIO_OUTPUTS])
    raw = raw.sort_values([CELL_COL, TIME_COL]).reset_index(drop=True)

    raw["x_m"], raw["y_m"] = latlon_to_local_xy(
        raw[LAT_COL], raw[LON_COL], base_lat, base_lon
    )

    prepared_parts = []
    diagnostics = []
    resample_rule = f"{RESAMPLE_MS}ms"

    for cell in ALL_CELLS:
        cell_raw = raw[raw[CELL_COL] == cell].copy().sort_values(TIME_COL)
        if cell_raw.empty:
            raise ValueError(f"Sheet1 içinde {cell} verisi bulunamadı.")

        # Uzun veri boşluklarının üzerinden GPS enterpolasyonu yapma.
        time_gap = cell_raw[TIME_COL].diff().dt.total_seconds().fillna(0.0)
        cell_raw["_segment_id"] = (time_gap > MAX_CONTINUOUS_GAP_S).cumsum()

        cell_parts = []
        total_control_points = 0
        total_removed_spikes = 0

        for _, segment in cell_raw.groupby("_segment_id", sort=True):
            segment = segment.sort_values(TIME_COL).copy()
            if segment.empty:
                continue

            radio = (
                segment.set_index(TIME_COL)[RADIO_OUTPUTS]
                .resample(resample_rule)
                .mean()
                .dropna(subset=RADIO_OUTPUTS)
            )
            if radio.empty:
                continue

            (
                x_interp,
                y_interp,
                lat_interp,
                lon_interp,
                n_control,
                n_removed,
            ) = interpolate_gps_to_times(
                segment,
                radio.index,
                base_lat,
                base_lon,
            )

            part = radio.copy()
            part["x_m"] = x_interp
            part["y_m"] = y_interp
            part[LAT_COL] = lat_interp
            part[LON_COL] = lon_interp
            part[CELL_COL] = cell
            part = part.reset_index()
            cell_parts.append(part)

            total_control_points += n_control
            total_removed_spikes += n_removed

        if not cell_parts:
            raise ValueError(f"{cell} için 1 saniyelik hazırlanmış veri üretilemedi.")

        cell_prepared = pd.concat(cell_parts, ignore_index=True)
        cell_prepared = cell_prepared.sort_values(TIME_COL).reset_index(drop=True)
        prepared_parts.append(cell_prepared)

        dt = cell_prepared[TIME_COL].diff().dt.total_seconds().to_numpy(dtype=float)
        dx = cell_prepared["x_m"].diff().to_numpy(dtype=float)
        dy = cell_prepared["y_m"].diff().to_numpy(dtype=float)
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        finite_speed = speed[np.isfinite(speed) & (dt > 0)]

        diagnostics.append({
            "Cell": cell,
            "Rows": len(cell_prepared),
            "GPS Control Points": total_control_points,
            "Removed GPS Spikes": total_removed_spikes,
            "Median Apparent Speed (m/s)": (
                float(np.median(finite_speed)) if len(finite_speed) else np.nan
            ),
            "P95 Apparent Speed (m/s)": (
                float(np.percentile(finite_speed, 95)) if len(finite_speed) else np.nan
            ),
            "Max Apparent Speed (m/s)": (
                float(np.max(finite_speed)) if len(finite_speed) else np.nan
            ),
        })

    data = pd.concat(prepared_parts, ignore_index=True)
    data = data.sort_values([CELL_COL, TIME_COL]).reset_index(drop=True)

    counts = data.groupby(CELL_COL).size().rename("1s Row Count")

    diagnostics_df = pd.DataFrame(diagnostics)

    return data, base_lat, base_lon, antenna_map, diagnostics_df


# =============================================================================
# 3B) ORTADAKİ UZUN R-Q-U TAKİP ROTASINI SEÇME
# =============================================================================

def select_long_central_rqu_track(data):
    """
    Görseldeki ortadaki uzun R-Q-U rotasını otomatik seçer.

    1) Sağ taraftaki kopuk noktaları x sınırıyla dışarıda bırakır.
    2) DBSCAN ile uzamsal olarak bağlı ana rota kümesini bulur.
    3) Zaman sırasındaki büyük zaman/konum sıçramalarını ayırır.
    4) R, Q ve U hücrelerini birlikte içeren en uzun sürekli parçayı seçer.
    """
    candidate = data[
        data[CELL_COL].isin(TRACK_CELLS)
        & (data["x_m"] <= TRACK_X_MAX_M)
    ].copy()

    if candidate.empty:
        raise ValueError("Merkezi R-Q-U takip rotası için aday veri bulunamadı.")

    xy = candidate[["x_m", "y_m"]].to_numpy(dtype=float)
    cluster_labels = DBSCAN(
        eps=TRACK_DBSCAN_EPS_M,
        min_samples=TRACK_DBSCAN_MIN_SAMPLES,
    ).fit_predict(xy)
    candidate["_spatial_cluster"] = cluster_labels

    valid = candidate[candidate["_spatial_cluster"] >= 0].copy()
    if valid.empty:
        raise ValueError(
            "DBSCAN merkezi rota kümesi oluşturamadı. "
            "TRACK_DBSCAN_EPS_M değerini artırmayı dene."
        )

    cluster_summary = (
        valid.groupby("_spatial_cluster")
        .agg(
            Rows=(CELL_COL, "size"),
            Cell_Count=(CELL_COL, "nunique"),
            Median_X=("x_m", "median"),
            Median_Y=("y_m", "median"),
        )
        .reset_index()
    )

    # Önce bütün R-Q-U hücrelerini içeren kümeleri tercih et; yoksa en büyük kümeyi al.
    all_cell_clusters = cluster_summary[
        cluster_summary["Cell_Count"] == len(TRACK_CELLS)
    ]
    if not all_cell_clusters.empty:
        chosen_cluster = int(
            all_cell_clusters.sort_values("Rows", ascending=False)
            .iloc[0]["_spatial_cluster"]
        )
    else:
        chosen_cluster = int(
            cluster_summary.sort_values("Rows", ascending=False)
            .iloc[0]["_spatial_cluster"]
        )

    spatial_track = (
        valid[valid["_spatial_cluster"] == chosen_cluster]
        .copy()
        .sort_values(TIME_COL)
        .drop_duplicates(subset=[TIME_COL], keep="last")
        .reset_index(drop=True)
    )

    if len(spatial_track) < 3:
        raise ValueError("Seçilen merkezi rota takip için çok kısa.")

    dt = spatial_track[TIME_COL].diff().dt.total_seconds()
    step = np.sqrt(
        spatial_track["x_m"].diff() ** 2
        + spatial_track["y_m"].diff() ** 2
    )
    new_segment = (
        dt.isna()
        | (dt > TRACK_MAX_TIME_GAP_S)
        | (dt < 0.0)
        | (step > TRACK_MAX_SPATIAL_JUMP_M)
    )
    spatial_track["_continuous_segment"] = new_segment.cumsum()

    segment_summary = (
        spatial_track.groupby("_continuous_segment")
        .agg(
            Rows=(CELL_COL, "size"),
            Cell_Count=(CELL_COL, "nunique"),
            Start_Time=(TIME_COL, "min"),
            End_Time=(TIME_COL, "max"),
        )
        .reset_index()
    )

    all_cell_segments = segment_summary[
        segment_summary["Cell_Count"] == len(TRACK_CELLS)
    ]
    if not all_cell_segments.empty:
        selected_segment = int(
            all_cell_segments.sort_values("Rows", ascending=False)
            .iloc[0]["_continuous_segment"]
        )
    else:
        # Zaman damgaları hücreler arasında ayrı kayıt bloklarıysa, uzamsal ana rotayı
        # koru. Bu fallback R-Q-U'nun ortadaki uzun geometrisini kaybetmez.
        selected_segment = None

    if selected_segment is None:
        track = spatial_track.copy()
    else:
        track = spatial_track[
            spatial_track["_continuous_segment"] == selected_segment
        ].copy()

    track = (
        track.sort_values(TIME_COL)
        .drop(columns=["_spatial_cluster", "_continuous_segment"], errors="ignore")
        .reset_index(drop=True)
    )

    if track[CELL_COL].nunique() < 2:
        raise ValueError(
            "Seçilen rota birden fazla sektörü içermiyor. "
            "TRACK_X_MAX_M veya DBSCAN ayarlarını kontrol et."
        )


    return track


# =============================================================================
# 4) R-Q-U RANDOM FOREST EĞİTİMİ, 70/30 - 80/20 - 90/10 KARŞILAŞTIRMASI
# =============================================================================

def split_each_cell_randomly(train_data, train_ratio, random_state):
    train_parts = []
    test_parts = []

    for i, cell in enumerate(TRAIN_CELLS):
        part = train_data[train_data[CELL_COL] == cell].copy()
        cell_train, cell_test = train_test_split(
            part,
            train_size=train_ratio,
            random_state=random_state + i,
            shuffle=True,
        )
        train_parts.append(cell_train)
        test_parts.append(cell_test)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Satır sırası modele herhangi bir düzen vermesin.
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def logical_output_metrics(y_true_internal, y_pred_internal, subset_name, split_label):
    rows = []

    for j, name in enumerate(RADIO_OUTPUTS):
        y_true = y_true_internal[:, j]
        y_pred = y_pred_internal[:, j]
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        scale = float(robust_range(y_true.reshape(-1, 1))[0])
        nrmse = rmse / scale
        r2 = r2_score(y_true, y_pred) if np.std(y_true) > 1e-12 else np.nan
        rows.append({
            "Split": split_label,
            "Subset": subset_name,
            "Output": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "NRMSE": nrmse,
        })

    return pd.DataFrame(rows)


def train_and_select_rf(train_data):

    # Aşırı derin/saf yapraklı ağaçları özellikle kullanmayan konservatif adaylar.
    # Her aday training kümesindeki OOB tahminleriyle seçilir; holdout test kümesi
    # yalnız seçilen adayın gerçek test performansını ölçmek için kullanılır.
    candidate_parameter_sets = [
        {
            "n_estimators": 120,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": 0.8,
            "max_samples": 0.80,
        },
        {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "max_features": 0.8,
            "max_samples": 0.85,
        },
        {
            "n_estimators": 180,
            "max_depth": 12,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": 1.0,
            "max_samples": 0.85,
        },
    ]

    split_summaries = []
    all_metric_tables = []
    experiment_objects = []
    candidate_rows = []

    for ratio_index, train_ratio in enumerate(TRAIN_RATIOS):
        train_percent = int(round(train_ratio * 100))
        split_label = f"{train_percent}/{100 - train_percent}"

        tr_df, te_df = split_each_cell_randomly(
            train_data,
            train_ratio=train_ratio,
            random_state=RANDOM_STATE + 100 * ratio_index,
        )

        x_train = tr_df[RF_INPUTS].to_numpy(dtype=float)
        y_train = tr_df[INTERNAL_OUTPUTS].to_numpy(dtype=float)
        x_test = te_df[RF_INPUTS].to_numpy(dtype=float)
        y_test = te_df[INTERNAL_OUTPUTS].to_numpy(dtype=float)

        best_candidate = None

        for candidate_no, params in enumerate(candidate_parameter_sets, start=1):
            candidate_model = RandomForestRegressor(
                **params,
                random_state=RANDOM_STATE + candidate_no,
                bootstrap=True,
                oob_score=True,
                n_jobs=N_JOBS,
            )
            candidate_model.fit(x_train, y_train)

            train_pred = candidate_model.predict(x_train)
            oob_pred = candidate_model.oob_prediction_

            train_candidate_metrics = logical_output_metrics(
                y_train, train_pred, "Candidate_Train", split_label
            )
            oob_candidate_metrics = logical_output_metrics(
                y_train, oob_pred, "Candidate_OOB", split_label
            )

            candidate_train_nrmse = float(train_candidate_metrics["NRMSE"].mean())
            candidate_oob_nrmse = float(oob_candidate_metrics["NRMSE"].mean())
            candidate_gap = max(0.0, candidate_oob_nrmse - candidate_train_nrmse)
            candidate_score = candidate_oob_nrmse + 0.25 * candidate_gap

            candidate_rows.append({
                "Split": split_label,
                "Candidate": candidate_no,
                "Train Mean NRMSE": candidate_train_nrmse,
                "OOB Mean NRMSE": candidate_oob_nrmse,
                "Overfit Gap": candidate_gap,
                "OOB Selection Score": candidate_score,
                "OOB R2": float(candidate_model.oob_score_),
                "Params": str(params),
            })

            if best_candidate is None or candidate_score < best_candidate["score"]:
                best_candidate = {
                    "score": candidate_score,
                    "params": dict(params),
                    "model": candidate_model,
                    "train_pred": train_pred,
                    "oob_nrmse": candidate_oob_nrmse,
                    "candidate_no": candidate_no,
                }

        model = best_candidate["model"]
        pred_train = best_candidate["train_pred"]
        pred_test = model.predict(x_test)

        train_metrics = logical_output_metrics(
            y_train, pred_train, "Train", split_label
        )
        test_metrics = logical_output_metrics(
            y_test, pred_test, "Test", split_label
        )
        metrics = pd.concat([train_metrics, test_metrics], ignore_index=True)
        all_metric_tables.append(metrics)

        train_mean_nrmse = float(train_metrics["NRMSE"].mean())
        test_mean_nrmse = float(test_metrics["NRMSE"].mean())
        overfit_gap = max(0.0, test_mean_nrmse - train_mean_nrmse)
        selection_score = test_mean_nrmse + 0.25 * overfit_gap

        summary = {
            "Split": split_label,
            "Train Ratio": train_ratio,
            "Train Rows": len(tr_df),
            "Test Rows": len(te_df),
            "Selected Candidate": best_candidate["candidate_no"],
            "Train Mean NRMSE": train_mean_nrmse,
            "OOB Mean NRMSE": best_candidate["oob_nrmse"],
            "Test Mean NRMSE": test_mean_nrmse,
            "Overfit Gap": overfit_gap,
            "Selection Score": selection_score,
            "OOB R2": float(model.oob_score_),
            "Best Params": str(best_candidate["params"]),
        }
        split_summaries.append(summary)

        experiment_objects.append({
            "summary": summary,
            "model": model,
            "train_df": tr_df,
            "test_df": te_df,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "pred_train": pred_train,
            "pred_test": pred_test,
            "best_params": best_candidate["params"],
        })


    summary_df = pd.DataFrame(split_summaries).sort_values(
        "Selection Score", ascending=True
    ).reset_index(drop=True)
    metrics_df = pd.concat(all_metric_tables, ignore_index=True)
    candidate_df = pd.DataFrame(candidate_rows)

    best_split_label = summary_df.iloc[0]["Split"]
    selected = next(
        obj for obj in experiment_objects if obj["summary"]["Split"] == best_split_label
    )


    selected_test_residuals = (
        selected["y_test"][:, :len(RADIO_OUTPUTS)]
        - selected["pred_test"][:, :len(RADIO_OUTPUTS)]
    )

    # Seçilen model yalnızca seçilen splitin eğitim bölümünde kalır.
    # Böylece %70/%80/%90 eğitim oranı korunur ve U verisinin yalnız seçilen
    # orandaki kısmı RF tarafından görülmüş olur.
    final_model = selected["model"]

    return (
        final_model,
        selected,
        selected_test_residuals,
        summary_df,
        metrics_df,
        candidate_df,
    )


# =============================================================================
# 5) R ÖLÇÜM KOVARYANSI
# =============================================================================

def build_measurement_covariance(test_residuals):

    if len(test_residuals) < 5:
        covariance = np.diag(MEASUREMENT_STD_FLOOR ** 2)
    else:
        covariance = LedoitWolf().fit(test_residuals).covariance_

    # Her özelliğin varyansı belirlenen tabanın altına düşmesin.
    current_diag = np.diag(covariance)
    floor_var = MEASUREMENT_STD_FLOOR ** 2
    additional_diag = np.maximum(floor_var - current_diag, 0.0)
    covariance = covariance + np.diag(additional_diag)

    # Seçilen splitin holdout artıklarından elde edilen kovaryans kullanılır.
    covariance = covariance * R_INFLATION
    covariance += np.eye(len(RADIO_OUTPUTS)) * 1e-6

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("R kovaryansı pozitif tanımlı oluşturulamadı.")

    r_inv = np.linalg.inv(covariance)

    return covariance, r_inv, logdet


def build_rf_measurement_grid(final_rf, xy_bounds, grid_step_m):
    """
    RF ölçüm fonksiyonunu düzenli bir XY ızgarasında bir kez hesaplar.
    PF sırasında binlerce kez RandomForest.predict çağırmak yerine hızlı
    bilinear interpolasyon kullanılır. Bu, aynı RF yüzeyinin sayısal
    hızlandırılmış temsilidir.
    """
    x_min, x_max, y_min, y_max = xy_bounds
    x_axis = np.arange(x_min, x_max + grid_step_m, grid_step_m, dtype=float)
    y_axis = np.arange(y_min, y_max + grid_step_m, grid_step_m, dtype=float)

    xx, yy = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])


    # Tek büyük predict çağrısı yerine makul chunk'lar bellek ve süreyi dengeler.
    chunk_size = 50_000
    prediction_chunks = []
    for start in range(0, len(grid_points), chunk_size):
        chunk = grid_points[start:start + chunk_size]
        prediction_chunks.append(
            final_rf.predict(chunk)[:, :len(RADIO_OUTPUTS)]
        )

    predictions = np.vstack(prediction_chunks)
    measurement_grid = predictions.reshape(
        len(y_axis), len(x_axis), len(RADIO_OUTPUTS)
    )
    return x_axis, y_axis, measurement_grid


def interpolate_measurement_grid(particle_xy, x_axis, y_axis, measurement_grid):
    """Parçacık konumlarında vektörize bilinear RF ölçüm interpolasyonu."""
    step_x = x_axis[1] - x_axis[0]
    step_y = y_axis[1] - y_axis[0]

    tx = (particle_xy[:, 0] - x_axis[0]) / step_x
    ty = (particle_xy[:, 1] - y_axis[0]) / step_y

    ix = np.floor(tx).astype(int)
    iy = np.floor(ty).astype(int)

    ix = np.clip(ix, 0, len(x_axis) - 2)
    iy = np.clip(iy, 0, len(y_axis) - 2)

    fx = np.clip(tx - ix, 0.0, 1.0)[:, None]
    fy = np.clip(ty - iy, 0.0, 1.0)[:, None]

    v00 = measurement_grid[iy, ix]
    v10 = measurement_grid[iy, ix + 1]
    v01 = measurement_grid[iy + 1, ix]
    v11 = measurement_grid[iy + 1, ix + 1]

    return (
        (1.0 - fx) * (1.0 - fy) * v00
        + fx * (1.0 - fy) * v10
        + (1.0 - fx) * fy * v01
        + fx * fy * v11
    )


# =============================================================================
# 6) CTRV PARTICLE FILTER
# =============================================================================

def run_particle_filter(final_rf, track_df, train_data, r_cov, r_inv, logdet_r):

    rng = np.random.default_rng(RANDOM_STATE)
    particles = create_initial_particles(track_df, rng)
    weights = np.full(N_PARTICLES, 1.0 / N_PARTICLES, dtype=float)

    # RF yalnız eğitim alanında anlamlıdır. R-Q-U eğitim sınırlarına güvenli bir marj ekle.
    x_min, x_max = train_data["x_m"].min(), train_data["x_m"].max()
    y_min, y_max = train_data["y_m"].min(), train_data["y_m"].max()
    margin = 150.0
    requested_bounds = (
        float(x_min - margin),
        float(x_max + margin),
        float(y_min - margin),
        float(y_max + margin),
    )

    x_grid, y_grid, rf_measurement_grid = build_rf_measurement_grid(
        final_rf, requested_bounds, RF_GRID_STEP_M
    )
    # Parçacık sınırlarını gerçek ızgara uçlarına eşitle.
    xy_bounds = (x_grid[0], x_grid[-1], y_grid[0], y_grid[-1])

    estimates = []
    covariance_diags = []
    n_eff_values = []
    resampled_flags = []

    previous_time = track_df[TIME_COL].iloc[0]

    for k, row in track_df.reset_index(drop=True).iterrows():
        current_time = row[TIME_COL]

        if k > 0:
            dt = (current_time - previous_time).total_seconds()
            ctrv_predict(particles, dt, rng, xy_bounds)
        previous_time = current_time

        # Her parçacığın konumundan RF ölçüm yüzeyi bilinear interpolasyonu.
        predicted_measurements = interpolate_measurement_grid(
            particles[:, :2], x_grid, y_grid, rf_measurement_grid
        )
        actual_measurement = row[RADIO_OUTPUTS].to_numpy(dtype=float)
        residuals = actual_measurement[None, :] - predicted_measurements

        log_likelihood = student_t_log_likelihood(
            residuals, r_inv, logdet_r, STUDENT_T_DOF
        )
        log_likelihood /= LIKELIHOOD_TEMPERATURE

        log_weights = np.log(weights + 1e-300) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weight_sum = np.sum(weights)

        if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
            weights.fill(1.0 / N_PARTICLES)
        else:
            weights /= weight_sum

        estimate = weighted_state_mean(particles, weights)
        covariance = state_covariance(particles, weights, estimate)
        n_eff = effective_sample_size(weights)

        estimates.append(estimate)
        covariance_diags.append(np.diag(covariance))
        n_eff_values.append(n_eff)

        resampled = False
        if n_eff < RESAMPLE_THRESHOLD * N_PARTICLES:
            indexes = systematic_resample(weights, rng)
            particles = particles[indexes].copy()
            particles += rng.normal(0.0, ROUGHENING_STD, size=particles.shape)
            particles[:, 3] = wrap_angle_rad(particles[:, 3])
            particles[:, 2] = np.clip(particles[:, 2], 0.0, 45.0)
            particles[:, 4] = np.clip(particles[:, 4], -1.2, 1.2)
            particles[:, 0] = np.clip(particles[:, 0], xy_bounds[0], xy_bounds[1])
            particles[:, 1] = np.clip(particles[:, 1], xy_bounds[2], xy_bounds[3])
            weights.fill(1.0 / N_PARTICLES)
            resampled = True

        resampled_flags.append(resampled)

    estimates = np.asarray(estimates)
    covariance_diags = np.asarray(covariance_diags)

    results = track_df.copy().reset_index(drop=True)
    results["pf_x_m"] = estimates[:, 0]
    results["pf_y_m"] = estimates[:, 1]
    results["pf_v_mps"] = estimates[:, 2]
    results["pf_theta_rad"] = estimates[:, 3]
    results["pf_theta_deg"] = wrap_angle_deg(np.rad2deg(estimates[:, 3]))
    results["pf_omega_rad_s"] = estimates[:, 4]
    results["pf_omega_deg_s"] = np.rad2deg(estimates[:, 4])
    results["N_eff"] = n_eff_values
    results["resampled"] = resampled_flags

    state_names = ["x", "y", "v", "theta", "omega"]
    for j, name in enumerate(state_names):
        results[f"pf_var_{name}"] = covariance_diags[:, j]

    results["position_error_m"] = np.sqrt(
        (results["pf_x_m"] - results["x_m"]) ** 2
        + (results["pf_y_m"] - results["y_m"]) ** 2
    )
    results["elapsed_s"] = (
        results[TIME_COL] - results[TIME_COL].iloc[0]
    ).dt.total_seconds()

    return results


# =============================================================================
# 7) SONUÇLAR VE YALNIZCA İKİ GRAFİK
# =============================================================================

def show_split_comparison_table(summary_df, selected_split):
    """RF split karşılaştırmasını terminal yerine tablo figürü olarak gösterir."""
    table_df = summary_df[
        [
            "Split",
            "Train Rows",
            "Test Rows",
            "Train Mean NRMSE",
            "OOB Mean NRMSE",
            "Test Mean NRMSE",
            "Overfit Gap",
            "Selection Score",
            "OOB R2",
        ]
    ].copy()

    display_df = table_df.copy()
    for col in [
        "Train Mean NRMSE",
        "OOB Mean NRMSE",
        "Test Mean NRMSE",
        "Overfit Gap",
        "Selection Score",
        "OOB R2",
    ]:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}")

    display_df = display_df.rename(columns={
        "Train Rows": "Train",
        "Test Rows": "Test",
        "Train Mean NRMSE": "Train NRMSE",
        "OOB Mean NRMSE": "OOB NRMSE",
        "Test Mean NRMSE": "Test NRMSE",
        "Overfit Gap": "Overfit",
        "Selection Score": "Skor",
        "OOB R2": "OOB R²",
    })

    fig, ax = plt.subplots(figsize=(13.5, 3.2))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    # Seçilen split satırını tabloda görsel olarak ayırt et.
    for row_index, split_value in enumerate(display_df["Split"], start=1):
        if split_value == selected_split:
            for col_index in range(len(display_df.columns)):
                table[(row_index, col_index)].set_facecolor("#fff2cc")

    for col_index in range(len(display_df.columns)):
        table[(0, col_index)].set_facecolor("#d9eaf7")
        table[(0, col_index)].set_text_props(weight="bold")

    ax.set_title(
        f"Random Forest Split Karşılaştırması - Seçilen Split: {selected_split}",
        pad=14,
        fontweight="bold",
    )
    plt.tight_layout()


def show_requested_plots(track_results, selected_split):
    """Yalnızca takip rotasını ve konum hatası CDF grafiğini gösterir."""

    # 1) Ortadaki uzun R-Q-U gerçek rota ve PF-CTRV tahmini
    plt.figure(figsize=(11, 8))
    plt.plot(
        track_results["x_m"],
        track_results["y_m"],
        color="black",
        linewidth=2.5,
        label="R-Q-U gerçek uzun rota",
        zorder=1,
    )

    cell_plot_styles = {
        "EOL8709R": ("red", "R gerçek noktaları"),
        "EOL8709Q": ("blue", "Q gerçek noktaları"),
        "EOL8709U": ("green", "U gerçek noktaları"),
    }
    for cell, (color, label) in cell_plot_styles.items():
        part = track_results[track_results[CELL_COL] == cell]
        if not part.empty:
            plt.scatter(
                part["x_m"], part["y_m"],
                s=22, color=color, label=label, zorder=3,
            )

    plt.plot(
        track_results["pf_x_m"],
        track_results["pf_y_m"],
        color="crimson",
        linestyle="--",
        linewidth=2.0,
        label="RF + PF-CTRV tahmini",
        zorder=2,
    )
    plt.scatter([0], [0], marker="^", s=180, label="Baz istasyonu")
    plt.xlabel("X (metre)")
    plt.ylabel("Y (metre)")
    plt.title(
        f"Uzun R-Q-U Rota Takibi: RF + CTRV Particle Filter "
        f"(Seçilen Train/Test: {selected_split})"
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    # 2) Konum hatası CDF
    err = np.sort(track_results["position_error_m"].to_numpy(dtype=float))
    cdf = np.arange(1, len(err) + 1) / len(err)
    mean_error = float(np.mean(err))
    p90_error = float(np.percentile(err, 90))

    plt.figure(figsize=(8, 6))
    plt.plot(err, cdf, linewidth=2.0)
    plt.axvline(mean_error, linestyle="--", label=f"Ortalama: {mean_error:.2f} m")
    plt.axvline(p90_error, linestyle=":", label=f"P90: {p90_error:.2f} m")
    plt.xlabel("Konum hatası (m)")
    plt.ylabel("Kümülatif olasılık")
    plt.title("Uzun R-Q-U Rota PF-CTRV Konum Hatası CDF")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    # Herhangi bir dosya kaydı yapılmaz.
    plt.show()


# =============================================================================
# 8) ANA PROGRAM
# =============================================================================

def main():
    np.set_printoptions(linewidth=180, precision=4, suppress=True)

    data, base_lat, base_lon, antenna_map, gps_diagnostics = load_and_prepare_data(
        EXCEL_PATH
    )

    # R, Q ve U hücrelerinin tamamı split deneylerine dâhildir.
    rf_data = data[data[CELL_COL].isin(TRAIN_CELLS)].copy().reset_index(drop=True)

    # PF, R-Q-U hücrelerinin görseldeki ortadaki uzun ve sürekli rotası üzerinde
    # değerlendirilir. Sağ taraftaki kopuk noktalar otomatik olarak dışarıda kalır.
    long_rqu_track = select_long_central_rqu_track(data)

    if rf_data.empty or long_rqu_track.empty:
        raise ValueError("RF eğitim verisi veya uzun R-Q-U takip rotası boş.")

    (
        final_rf,
        selected,
        test_residuals,
        summary_df,
        metrics_df,
        candidate_df,
    ) = train_and_select_rf(rf_data)

    show_split_comparison_table(
        summary_df=summary_df,
        selected_split=selected["summary"]["Split"],
    )

    r_cov, r_inv, logdet_r = build_measurement_covariance(test_residuals)

    # Izgara sınırları yalnız seçilen splitin gerçek eğitim bölümünden belirlenir.
    selected_train_data = selected["train_df"].copy()

    track_results = run_particle_filter(
        final_rf=final_rf,
        track_df=long_rqu_track,
        train_data=selected_train_data,
        r_cov=r_cov,
        r_inv=r_inv,
        logdet_r=logdet_r,
    )


    show_requested_plots(
        track_results=track_results,
        selected_split=selected["summary"]["Split"],
    )


if __name__ == "__main__":
    main()
