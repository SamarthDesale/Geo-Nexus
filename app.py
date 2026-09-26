import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
import os
from datetime import date
from scipy.ndimage import label

# ---------------------------------------------------------------------------
# 1. Earth Engine Initialization with Direct Project ID
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Geo-Nexus | Earth Engine Change Detection",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

EE_PROJECT_ID = "project-7997c97c-2387-40e2-804"

@st.cache_resource
def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT_ID)

init_ee()

# ---------------------------------------------------------------------------
# 2. Geo-Nexus Architecture Definition
# ---------------------------------------------------------------------------
class SplitStem(nn.Module):
    def __init__(self, n_raw=11, n_derived=2, out_ch=64, use_derived=True):
        super().__init__()
        self.use_derived = use_derived
        self.raw = nn.Conv2d(n_raw, out_ch, 7, stride=2, padding=3, bias=False)
        self.derived = nn.Conv2d(n_derived, out_ch, 7, stride=2, padding=3, bias=False)
        nn.init.zeros_(self.derived.weight)

    def forward(self, x_raw, x_derived):
        y = self.raw(x_raw)
        if self.use_derived:
            y = y + self.derived(x_derived)
        return y

class BranchEncoder(nn.Module):
    def __init__(self, n_raw, n_derived, use_derived=True):
        super().__init__()
        r = resnet18(weights=None)
        self.stem = SplitStem(n_raw, n_derived, 64, use_derived)
        self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.layer3, self.layer4 = r.layer3, r.layer4

    def forward(self, x_raw, x_derived):
        x = self.maxpool(self.relu(self.bn1(self.stem(x_raw, x_derived))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class DualEncoder(nn.Module):
    def __init__(self, use_derived=True):
        super().__init__()
        self.optical = BranchEncoder(11, 2, use_derived)
        self.sar = BranchEncoder(3, 1, use_derived=False)

    def forward(self, x):
        f_opt = self.optical(x[:, 0:11], x[:, 11:13])
        zeros = torch.zeros(x.shape[0], 1, *x.shape[2:], device=x.device, dtype=x.dtype)
        f_sar = self.sar(x[:, 13:16], zeros)
        return f_opt, f_sar

class QualityGate(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        self.gates = nn.ModuleList([nn.Conv2d(2 * c + 1, 1, kernel_size=1) for c in channels])
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, 2.0)
        self.mode = 'gated'

    def forward(self, f_opt, f_sar, q):
        fused, gates = [], []
        for i, (fo, fs) in enumerate(zip(f_opt, f_sar)):
            if self.mode == 'optical_only':
                fused.append(fo)
                gates.append(torch.ones_like(fo[:, :1]))
                continue
            qs = F.adaptive_avg_pool2d(q, fo.shape[-2:])
            g = torch.sigmoid(self.gates[i](torch.cat([fo, fs, qs], dim=1)))
            fused.append(g * fo + (1.0 - g) * fs)
            gates.append(g)
        return fused, gates

class TemporalDiff(nn.Module):
    def forward(self, f1_list, f2_list):
        out = []
        for f1, f2 in zip(f1_list, f2_list):
            d_abs = (f1 - f2).abs()
            d_mul = f1 * f2
            d_cos = F.cosine_similarity(f1, f2, dim=1).unsqueeze(1)
            out.append(torch.cat([d_abs, d_mul, d_cos], dim=1))
        return out

class LKABlock(nn.Module):
    def __init__(self, dim, k=7, d=2):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.p1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        dw_k = 2 * d - 1
        dwd_k = 3
        self.dw = nn.Conv2d(dim, dim, dw_k, padding=dw_k // 2, groups=dim)
        self.dwd = nn.Conv2d(dim, dim, dwd_k, padding=(dwd_k // 2) * d, groups=dim, dilation=d)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.p2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x
        x = self.act(self.p1(self.norm(x)))
        attn = self.pw(self.dwd(self.dw(x)))
        return u + self.p2(x * attn)

def conv_bn_gelu(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.GELU())

class LKADecoder(nn.Module):
    def __init__(self, in_ch=(129, 257, 513, 1025), dec_ch=(256, 128, 64), k=7):
        super().__init__()
        c1, c2, c3 = dec_ch
        self.bottleneck = nn.Sequential(conv_bn_gelu(in_ch[3], c1, 1), LKABlock(c1, k), LKABlock(c1, k))
        self.skip3 = conv_bn_gelu(in_ch[2], c1, 1)
        self.skip2 = conv_bn_gelu(in_ch[1], c2, 1)
        self.skip1 = conv_bn_gelu(in_ch[0], c3, 1)
        self.up3 = nn.Sequential(conv_bn_gelu(c1 + c1, c1), LKABlock(c1, k))
        self.up2 = nn.Sequential(conv_bn_gelu(c1 + c2, c2), LKABlock(c2, k))
        self.up1 = nn.Sequential(conv_bn_gelu(c2 + c3, c3), LKABlock(c3, k))
        self.head3 = nn.Conv2d(c1, 1, 1)
        self.head2 = nn.Conv2d(c2, 1, 1)
        self.head1 = nn.Conv2d(c3, 1, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, d):
        x = self.bottleneck(d[3])
        s3 = self.skip3(d[2])
        x = self.up3(torch.cat([self._up(x, s3), s3], dim=1))
        o3 = self.head3(x)
        s2 = self.skip2(d[1])
        x = self.up2(torch.cat([self._up(x, s2), s2], dim=1))
        o2 = self.head2(x)
        s1 = self.skip1(d[0])
        x = self.up1(torch.cat([self._up(x, s1), s1], dim=1))
        o1 = self.head1(x)
        return o1, o2, o3

class GeoNexusCD(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualEncoder(use_derived=True)
        self.gate = QualityGate((64, 128, 256, 512))
        self.tdiff = TemporalDiff()
        self.decoder = LKADecoder()

    def set_mode(self, mode):
        self.gate.mode = mode
        return self

    def forward(self, x1, x2):
        fo1, fs1 = self.encoder(x1)
        fo2, fs2 = self.encoder(x2)
        f1, g1 = self.gate(fo1, fs1, x1[:, 16:17])
        f2, g2 = self.gate(fo2, fs2, x2[:, 16:17])
        d = self.tdiff(f1, f2)
        o1, o2, o3 = self.decoder(d)
        up = lambda o: F.interpolate(o, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        return {'logits': up(o1), 'aux8': up(o2), 'aux16': up(o3), 'gates': g1 + g2}

# ---------------------------------------------------------------------------
# 3. Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_geonexus_model(weights_path="mh_fewshot_reweighted.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeoNexusCD().to(device)
    
    if not os.path.exists(weights_path):
        for root, dirs, files in os.walk('.'):
            if 'mh_fewshot_reweighted.pth' in files:
                weights_path = os.path.join(root, 'mh_fewshot_reweighted.pth')
                break
                
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, device

# ---------------------------------------------------------------------------
# 4. GEE Live Data Pipeline (Mask-Safe Median Mosaic)
# ---------------------------------------------------------------------------
def fetch_satellite_data(lon, lat, start_date, end_date):
    """Pulls aligned 17-channel Sentinel-2 / Sentinel-1 array from Earth Engine safely."""
    point = ee.Geometry.Point([lon, lat])
    utm_zone = int((lon + 180) / 6) + 1
    crs = f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"
    roi = point.buffer(640).bounds()

    # 1. Optical: Sentinel-2 L2A Harmonized
    # Use median composite over low-cloud scenes to eliminate tile edge / missing pixel masks
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(roi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35)))
    
    # Fallback to unrestricted if cloudy filter yields empty collection
    s2_img = ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_col.median(),
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(roi)
          .filterDate(start_date, end_date)
          .median()
    )
    s2 = ee.Image(s2_img)

    opt_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    s2_bands = s2.select(opt_bands).divide(10000.0).toFloat()
    
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi').toFloat()
    ndbi = s2.normalizedDifference(['B11', 'B8']).rename('ndbi').toFloat()
    
    # Safe Quality map
    q_map = ee.Image.constant(1.0).rename('q').toFloat()

    # 2. SAR: Sentinel-1 GRD IW
    s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
              .filterBounds(roi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
              .filter(ee.Filter.eq('instrumentMode', 'IW')))
              
    s1 = ee.Image(ee.Algorithms.If(
        s1_col.size().gt(0),
        s1_col.mean(),
        ee.Image.constant(-15.0).rename('VV').addBands(ee.Image.constant(-22.0).rename('VH'))
    ))
    
    vv = s1.select('VV').rename('vv').toFloat()
    vh = s1.select('VH').rename('vh').toFloat()
    ratio = vv.subtract(vh).rename('ratio').toFloat()

    # 3. Concatenate and explicitly unmask all channels to 0.0
    composite = ee.Image.cat([s2_bands, ndvi, ndbi, vv, vh, ratio, q_map]).toFloat()
    composite = composite.unmask(0.0)
    composite = composite.reproject(crs=crs, scale=10)

    # Supply defaultValue=0.0 to sampleRectangle so empty pixels never crash GEE
    pixels = composite.sampleRectangle(region=roi, defaultValue=0.0).getInfo()
    
    arr_channels = []
    keys = opt_bands + ['ndvi', 'ndbi', 'vv', 'vh', 'ratio', 'q']
    for k in keys:
        raw_ch = np.array(pixels['properties'][k], dtype=np.float32)
        h, w = raw_ch.shape
        
        # Center-crop or trim
        if h > 128:
            start_y = (h - 128) // 2
            raw_ch = raw_ch[start_y:start_y+128, :]
        if w > 128:
            start_x = (w - 128) // 2
            raw_ch = raw_ch[:, start_x:start_x+128]
            
        if raw_ch.shape != (128, 128):
            pad_h = max(0, 128 - raw_ch.shape[0])
            pad_w = max(0, 128 - raw_ch.shape[1])
            raw_ch = np.pad(raw_ch, ((0, pad_h), (0, pad_w)), mode='edge')
            raw_ch = raw_ch[:128, :128]
            
        arr_channels.append(raw_ch)
        
    return np.stack(arr_channels, axis=0).astype(np.float32)
# ---------------------------------------------------------------------------
# 5. Calibrated Step 6 Physical Spectral Engine
# ---------------------------------------------------------------------------
COLOURS = {
    0: (0, 0, 0),        # Unchanged
    1: (30, 100, 220),   # Water Gain (Blue)
    2: (80, 200, 220),   # Water Loss (Cyan)
    3: (255, 200, 0),    # Construction / Urban (Yellow)
    4: (220, 50, 50),    # Vegetation Loss (Red)
    5: (50, 180, 50),    # Vegetation Gain (Green)
    6: (180, 180, 180)   # Other Dynamic Changes (Grey)
}

def spectral_typing(t1, t2, binary_mask):
    green1, swir1_1 = t1[2], t1[9]
    green2, swir1_2 = t2[2], t2[9]
    
    mndwi1 = (green1 - swir1_1) / (green1 + swir1_1 + 1e-6)
    mndwi2 = (green2 - swir1_2) / (green2 + swir1_2 + 1e-6)
    d_mndwi = mndwi2 - mndwi1
    
    d_ndvi = t2[11] - t1[11]
    d_ndbi = t2[12] - t1[12]

    typed = np.zeros_like(binary_mask, dtype=np.uint8)
    m = binary_mask.astype(bool)
    
    typed[m & (d_mndwi > 0.12)] = 1                      # Water Inundation / Gain (Blue)
    typed[m & (typed == 0) & (d_mndwi < -0.12)] = 2      # Water Loss / Shoreline Exposed (Cyan)
    typed[m & (typed == 0) & (d_ndbi > 0.10)] = 3        # Construction / Built-up (Yellow)
    typed[m & (typed == 0) & (d_ndvi < -0.15)] = 4       # Significant Vegetation Loss (Red)
    typed[m & (typed == 0) & (d_ndvi > 0.15)] = 5        # Significant Vegetation Gain (Green)
    typed[m & (typed == 0)] = 6                          # Other Complex Dynamics (Grey)
    return typed

# ---------------------------------------------------------------------------
# 6. Streamlit User Interface - Vibrant Mixed-Color Dashboard
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    /* Vibrant Mixed Gradient CTA Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(219, 39, 119, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

# Top Mixed-Color Header Card
with st.container(border=True):
    head_left, head_right = st.columns([3, 1])
    with head_left:
        st.markdown("<h2 style='margin:0; background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🛰️ Geo-Nexus</h2>", unsafe_allow_html=True)
        st.caption("Multimodal Earth Engine Change Detection • Dual-Encoder (Optical + SAR) LKA Architecture")
    with head_right:
        st.markdown("**`v2.4 Live`** &nbsp;•&nbsp; 🟢 **GEE Active**")
        st.caption(f"Project: `{EE_PROJECT_ID}`")

# Session state initialization for interactive map clicks
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 18.9905
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = 73.0725

location_presets = {
    "Custom / Map-Selected Coordinates": None,
    "Navi Mumbai Airport (Heavy Construction)": (18.9905, 73.0725),
    "Khadakwasla Dam Shoreline (Water Dynamics)": (18.4350, 73.7620),
    "Samruddhi Expressway Interchange (Road Infrastructure)": (19.8970, 74.4750),
    "Dhule MIDC (Industrial Growth)": (20.9125, 74.7410)
}

def apply_preset():
    choice = st.session_state.preset_choice
    if location_presets[choice] is not None:
        p_lat, p_lon = location_presets[choice]
        st.session_state.selected_lat = p_lat
        st.session_state.selected_lon = p_lon

# Sidebar Configuration (Clean full-width inputs without expander glitches)
st.sidebar.subheader("📍 Target Location")
st.sidebar.selectbox("Test Location Preset", list(location_presets.keys()), key="preset_choice", on_change=apply_preset)

lat = st.sidebar.number_input("Center Latitude", value=float(st.session_state.selected_lat), format="%.5f", key="input_lat")
lon = st.sidebar.number_input("Center Longitude", value=float(st.session_state.selected_lon), format="%.5f", key="input_lon")

st.session_state.selected_lat = lat
st.session_state.selected_lon = lon

st.sidebar.divider()
st.sidebar.subheader("🗺️ Basemap Tile Layer")
map_style = st.sidebar.radio("Basemap Style", ["OpenStreetMap", "Esri Satellite Imagery"])

st.sidebar.divider()
st.sidebar.subheader("📅 Temporal Windows")
st.sidebar.caption("T1 (Historical Baseline)")
t1_start = st.sidebar.date_input("T1 Start Date", value=date(2020, 1, 1))
t1_end = st.sidebar.date_input("T1 End Date", value=date(2020, 5, 30))

st.sidebar.caption("T2 (Recent Monitoring)")
t2_start = st.sidebar.date_input("T2 Start Date", value=date(2024, 1, 1))
t2_end = st.sidebar.date_input("T2 End Date", value=date(2024, 5, 30))

st.sidebar.divider()
st.sidebar.subheader("🔬 Model Parameters")
fusion_mode = st.sidebar.selectbox("Inference Mode", ["gated", "optical_only"])
thr_slider = st.sidebar.slider("Detection Sensitivity (Cutoff)", min_value=0.10, max_value=0.85, value=0.40, step=0.05,
                               help="Higher threshold = less sensitive (fewer false alarms). Recommended: 0.35 - 0.50")
min_area_filter = st.sidebar.slider("Noise Speckle Filter (px)", min_value=1, max_value=25, value=6, step=1,
                                    help="Removes isolated salt-and-pepper pixel noise.")

# Interactive Leaflet Map Card
curr_lat, curr_lon = st.session_state.selected_lat, st.session_state.selected_lon

with st.container(border=True):
    st.subheader("🗺️ Target Region Selection")
    st.caption("Click anywhere on the map to re-center the 1280m × 1280m inference receptive field.")
    
    # Clean, key-free tile layer configuration
    if map_style == "OpenStreetMap":
        m = folium.Map(location=[curr_lat, curr_lon], zoom_start=14, tiles="OpenStreetMap")
    else:
        m = folium.Map(
            location=[curr_lat, curr_lon],
            zoom_start=14,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery"
        )

    # Draw 1280m x 1280m inference box preview (0.0057 deg lat ~ 640m radius)
    d_lat = 640.0 / 111320.0
    d_lon = 640.0 / (111320.0 * np.cos(np.radians(curr_lat)))
    bounds = [[curr_lat - d_lat, curr_lon - d_lon], [curr_lat + d_lat, curr_lon + d_lon]]

    folium.Rectangle(
        bounds=bounds,
        color="#2563EB",
        weight=2,
        fill=True,
        fill_color="#3B82F6",
        fill_opacity=0.18,
        popup="128x128 Model Receptive Field (1280m x 1280m)"
    ).add_to(m)

    folium.Marker(
        [curr_lat, curr_lon],
        popup=f"Selected Center: ({curr_lat:.4f}, {curr_lon:.4f})",
        tooltip="Active Inference Center",
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa")
    ).add_to(m)

    # Capture clicks from the map with robust width
    map_output = st_folium(m, height=420, use_container_width=True, returned_objects=["last_clicked"])

    if map_output and map_output.get("last_clicked"):
        click = map_output["last_clicked"]
        click_lat = round(click["lat"], 5)
        click_lon = round(click["lng"], 5)
        
        if click_lat != st.session_state.selected_lat or click_lon != st.session_state.selected_lon:
            st.session_state.selected_lat = click_lat
            st.session_state.selected_lon = click_lon
            st.rerun()

    st.info(f"📍 **Active Target Center:** Latitude `{st.session_state.selected_lat:.5f}`, Longitude `{st.session_state.selected_lon:.5f}`")

# Primary CTA Action Button
st.write("")
if st.button("🚀 Run Live Change Detection", type="primary", use_container_width=True):
    with st.spinner("Streaming multispectral optical and C-band SAR directly from GEE..."):
        try:
            # 1. Fetch raw arrays
            t1_arr = fetch_satellite_data(st.session_state.selected_lon, st.session_state.selected_lat, str(t1_start), str(t1_end))
            t2_arr = fetch_satellite_data(st.session_state.selected_lon, st.session_state.selected_lat, str(t2_start), str(t2_end))
            
            # 2. SAR scaling
            t1_proc = t1_arr.copy()
            t2_proc = t2_arr.copy()
            t1_proc[13:16] *= 100.0
            t2_proc[13:16] *= 100.0

            # 3. JOINT PER-CHANNEL STANDARDIZATION (Eliminates seasonal false positive triggers)
            for ch in range(17):
                if ch == 16:
                    continue
                joint_mean = 0.5 * (np.nanmean(t1_proc[ch]) + np.nanmean(t2_proc[ch]))
                joint_std = 0.5 * (np.nanstd(t1_proc[ch]) + np.nanstd(t2_proc[ch])) + 1e-6
                t1_proc[ch] = (t1_proc[ch] - joint_mean) / joint_std
                t2_proc[ch] = (t2_proc[ch] - joint_mean) / joint_std

            t1_norm = np.nan_to_num(t1_proc, nan=0.0, posinf=0.0, neginf=0.0)
            t2_norm = np.nan_to_num(t2_proc, nan=0.0, posinf=0.0, neginf=0.0)
            
            t1_tensor = torch.from_numpy(t1_norm).unsqueeze(0).float()
            t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).float()
            
            # 4. Model Inference
            model, dev = load_geonexus_model("mh_fewshot_reweighted.pth")
            model.set_mode(fusion_mode)
            
            with torch.no_grad():
                out = model(t1_tensor.to(dev), t2_tensor.to(dev))
                prob_map = torch.sigmoid(out['logits']).cpu().numpy()[0, 0].copy()
                prob_map = np.nan_to_num(prob_map, nan=0.0)
                
                bin_mask = (prob_map >= thr_slider).astype(np.uint8)
                
                # Zero out 2-pixel spatial border artifacts
                bin_mask[:, :2] = 0
                bin_mask[:, -2:] = 0
                bin_mask[:2, :] = 0
                bin_mask[-2:, :] = 0
                
                # Speckle / Noise Removal Filter
                if min_area_filter > 1:
                    labeled, num_features = label(bin_mask)
                    for feat in range(1, num_features + 1):
                        coords = (labeled == feat)
                        if np.sum(coords) < min_area_filter:
                            bin_mask[coords] = 0

            # 5. Step 6 Physical Typing
            typed_map = spectral_typing(t1_arr, t2_arr, bin_mask)
            
            # Natural Auto-Stretched True-Color RGB: B4 (Red), B3 (Green), B2 (Blue)
            def make_clean_rgb(arr):
                rgb = np.stack([arr[3], arr[2], arr[1]], axis=-1)
                p2, p98 = np.percentile(rgb, (2, 98))
                rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)
                return rgb
                
            t1_rgb = make_clean_rgb(t1_arr)
            t2_rgb = make_clean_rgb(t2_arr)
            
            rgb_spectral = np.zeros((128, 128, 3), dtype=np.uint8)
            for c_id, color in COLOURS.items():
                rgb_spectral[typed_map == c_id] = color
                
            st.toast("Analysis Complete!", icon="✅")
            
            # Surface Dynamics Statistics Calculations
            total_px = 128 * 128
            chg_px = int(np.sum(bin_mask))
            chg_pct = (chg_px / total_px) * 100.0
            
            built_up_px = int((typed_map == 3).sum())
            water_gain_px = int((typed_map == 1).sum())
            water_loss_px = int((typed_map == 2).sum())
            veg_gain_px = int((typed_map == 5).sum())
            veg_loss_px = int((typed_map == 4).sum())
            other_px = int((typed_map == 6).sum())
            
            class_counts = {
                "Construction / Built-up": built_up_px,
                "Water Inundation": water_gain_px,
                "Water Loss / Drought": water_loss_px,
                "Vegetation Growth": veg_gain_px,
                "Vegetation Clearing": veg_loss_px,
                "Other Dynamics": other_px
            }
            dominant_name = max(class_counts, key=class_counts.get) if chg_px > 0 else "Stable / No Change"
            
            # Mixed-Color Metric Cards Row
            with st.container(border=True):
                st.subheader("📊 Change Detection Summary")
                st.caption(f"Analysis between T1 ({t1_start}) and T2 ({t2_start}) across 1.64 km² receptive field")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🔵 Monitored Area", "1.64 km²", "16,384 px")
                m2.metric("🔴 Detected Change", f"{chg_pct:.2f}%", f"{chg_px:,} px")
                m3.metric("🟡 Dominant Dynamic", dominant_name)
                m4.metric("🟢 Sensitivity", f"t ≥ {thr_slider:.2f}", f"Speckle: {min_area_filter}px")
            
            # Visual Deliverables inside Modern Mixed Tabs
            vis_tabs = st.tabs(["🖼️ Side-by-Side Comparison", "🔲 4-Panel Detail Grid"])
            
            with vis_tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    with st.container(border=True):
                        st.markdown("<span style='color: #60a5fa; font-weight: 600;'>🟦 T1 Optical (Historical Baseline)</span>", unsafe_allow_html=True)
                        st.image(t1_rgb, use_container_width=True)
                    with st.container(border=True):
                        st.markdown(f"<span style='color: #f59e0b; font-weight: 600;'>🟨 Dual-Encoder Binary Change Mask (t={thr_slider:.2f})</span>", unsafe_allow_html=True)
                        st.image(bin_mask * 255, use_container_width=True)
                with col2:
                    with st.container(border=True):
                        st.markdown("<span style='color: #c084fc; font-weight: 600;'>🟪 T2 Optical (Recent Monitoring)</span>", unsafe_allow_html=True)
                        st.image(t2_rgb, use_container_width=True)
                    with st.container(border=True):
                        st.markdown("<span style='color: #f472b6; font-weight: 600;'>🌈 Semantic Typing (Physical Dynamics)</span>", unsafe_allow_html=True)
                        st.image(rgb_spectral, use_container_width=True)
            
            with vis_tabs[1]:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    with st.container(border=True):
                        st.markdown("**🔵 T1 Baseline**")
                        st.image(t1_rgb, use_container_width=True)
                with c2:
                    with st.container(border=True):
                        st.markdown("**🟣 T2 Recent**")
                        st.image(t2_rgb, use_container_width=True)
                with c3:
                    with st.container(border=True):
                        st.markdown("**🟡 Binary Mask**")
                        st.image(bin_mask * 255, use_container_width=True)
                with c4:
                    with st.container(border=True):
                        st.markdown("**🌈 Semantic Map**")
                        st.image(rgb_spectral, use_container_width=True)
                    
            # Semantic Distribution Section with Harmonious Mixed Colors
            with st.container(border=True):
                st.subheader("🌐 Surface Dynamics Classification")
                st.caption("Detailed physical spectral distribution across detected change pixels")
                
                sem_c1, sem_c2, sem_c3 = st.columns(3)
                with sem_c1:
                    st.metric("🟡 Construction / Built-up", f"{built_up_px:,} px", f"{(built_up_px / total_px * 100):.2f}%")
                    st.metric("🔵 Water Inundation / Gain", f"{water_gain_px:,} px", f"{(water_gain_px / total_px * 100):.2f}%")
                with sem_c2:
                    st.metric("🌐 Water Loss / Drought", f"{water_loss_px:,} px", f"{(water_loss_px / total_px * 100):.2f}%")
                    st.metric("🟢 Vegetation Growth", f"{veg_gain_px:,} px", f"{(veg_gain_px / total_px * 100):.2f}%")
                with sem_c3:
                    st.metric("🔴 Vegetation Loss / Clearing", f"{veg_loss_px:,} px", f"{(veg_loss_px / total_px * 100):.2f}%")
                    st.metric("⚪ Other Dynamics", f"{other_px:,} px", f"{(other_px / total_px * 100):.2f}%")

        except Exception as e:
            st.error(f"Error executing Earth Engine inference pipeline: {str(e)}")