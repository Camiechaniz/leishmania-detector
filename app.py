import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import time
import matplotlib.pyplot as plt
import gc
import zipfile
import json

st.set_page_config(page_title="Detector Leishmania y Macrófagos", page_icon="Microbe", layout="wide")

# ==================== RUTAS ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos"
MACROFAGOS_PATH = MODEL_DIR / "best_macrofagos.pt"
PARASITOS_PATH = MODEL_DIR / "best_parasitos.pt"

# Carpeta de salida con fecha/hora
OUTPUT_DIR = BASE_DIR / "salidas" / time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORIAL_CSV = BASE_DIR / "HISTORIAL_TODOS_DATASETS.csv"
GRAFICO_DATA_GLOBAL = BASE_DIR / "grafico_data_global.json"

# ==================== PARÁMETROS ====================
BORDER_DRAW_PX = 6
TOUCH_MARGIN_PX = 12
INPAINT_RADIUS = 60
FEATHER_GAUSS = 51
FEATHER_SIGMA = 25
YOLO_SIZE = 1024

CONF_MACROFAGOS = 0.75
CONF_PARASITOS = 0.50

DEFAULT_RECT_W = 3000
DEFAULT_RECT_H = 2000

# ==================== CARGAR MODELOS ====================
_model_macrofagos = None
def get_model_macrofagos():
    global _model_macrofagos
    if _model_macrofagos is None:
        _model_macrofagos = YOLO(MACROFAGOS_PATH)
    return _model_macrofagos

_model_parasitos = None
def get_model_parasitos():
    global _model_parasitos
    if _model_parasitos is None:
        _model_parasitos = YOLO(PARASITOS_PATH)
    return _model_parasitos

# ==================== UTILIDADES ====================
def imread_robusto(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    pil = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def bgr_to_hsi_s_channel(bgr):
    bgr = bgr.astype(np.float32) / 255.0
    B, G, R = cv2.split(bgr)
    eps = 1e-6
    S = 1 - (3.0 / (R + G + B + eps)) * np.minimum(np.minimum(R, G), B)
    return (np.clip(S, 0, 1) * 255).astype(np.uint8)

# ==================== LETTERBOX OFICIAL YOLOv8 ====================
def letterbox_any(img, new_shape=1024, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def unletterbox_boxes(boxes, ratio, pad):
    dw, dh = pad
    r_w, r_h = ratio
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r_h
    return boxes

# ==================== MÁSCARAS Y BORDES ====================
def build_mask_white_auto(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    dL = (255.0 - L).astype(np.float32)
    dA = (A.astype(np.float32) - 128.0)
    dB = (B.astype(np.float32) - 128.0)
    deltaE = np.sqrt(dL*dL + dA*dA + dB*dB)
    _, th = cv2.threshold(np.clip(deltaE, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    mask_de = (deltaE > th * 0.95).astype(np.uint8) * 255
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]
    _, th_s = cv2.threshold(S, 0, 255, cv2.THRESH_OTSU)
    mask_s = (S > th_s * 0.95).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_de, mask_s)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def apply_mask_on_S(S_gray, mask):
    if mask is None:
        return S_gray
    if mask.shape[:2] != S_gray.shape[:2]:
        mask = cv2.resize(mask, (S_gray.shape[1], S_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    Sinp = cv2.inpaint(S_gray, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    mask_dil = cv2.dilate(mask, np.ones((5,5)), iterations=2)
    mask_blur = cv2.GaussianBlur(mask_dil, (FEATHER_GAUSS, FEATHER_GAUSS), FEATHER_SIGMA).astype(np.float32)/255
    soft_bg = cv2.GaussianBlur(Sinp, (21,21), 10)
    return (Sinp.astype(np.float32)*(1-mask_blur) + soft_bg.astype(np.float32)*mask_blur).astype(np.uint8)

def make_rect_edge_mask(H, W, margin=TOUCH_MARGIN_PX):
    outer = np.full((H, W), 255, np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2*margin+1, 2*margin+1))
    inner = cv2.erode(outer, k)
    return cv2.subtract(outer, inner)

# ==================== FONDO NEGRO CIRCULAR ====================
def circular_fondo_negro_y_borde(img_bgr, draw_px=BORDER_DRAW_PX, touch_margin_px=TOUCH_MARGIN_PX, circle_grow_px=4):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def circularity(cnt):
        a = cv2.contourArea(cnt); p = cv2.arcLength(cnt, True)
        return 0 if p == 0 else 4*np.pi*a/(p*p)

    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.01*W*H: continue
        circ = circularity(c)
        M = cv2.moments(c)
        cx = M["m10"]/M["m00"] if M["m00"] > 0 else 0
        cy = M["m01"]/M["m00"] if M["m00"] > 0 else 0
        center_norm = np.hypot(cx - W/2, cy - H/2) / (np.hypot(W/2, H/2) + 1e-6)
        score = 0.7*(area/(W*H)) + 0.2*circ + 0.1*(1.0 - center_norm)
        if best is None or score > best[0]:
            best = (score, c)

    if best is None:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(H,W)//6,
                                   param1=120, param2=40, minRadius=min(H,W)//7, maxRadius=min(H,W)//2)
        if circles is not None:
            cx, cy, r = np.uint16(np.around(circles[0][np.argmax(circles[0][:,2])]))
        else:
            cx, cy, r = W//2, H//2, min(H,W)//3
    else:
        (cx_f, cy_f), r_f = cv2.minEnclosingCircle(best[1])
        cx, cy, r = int(round(cx_f)), int(round(cy_f)), int(round(r_f))

    r = int(r) + circle_grow_px
    r = max(5, min(r, cx, cy, W-1-cx, H-1-cy))

    mask_bin = np.zeros((H,W), np.uint8)
    cv2.circle(mask_bin, (cx, cy), r, 255, -1)
    result_bg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_bin)

    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*touch_margin_px+1, 2*touch_margin_px+1))
    inner = cv2.erode(mask_bin, k_ring, iterations=1)
    edge_mask = cv2.subtract(mask_bin, inner)

    result_edge = result_bg.copy()
    cv2.circle(result_edge, (cx, cy), r, (0,0,255), thickness=draw_px)

    return result_bg, result_edge, mask_bin, edge_mask, (cx, cy), r

# ==================== DETECCIÓN PRINCIPAL ====================
def detect_on_S_with_border(S_gray, edge_mask, conf_macro, conf_para):
    H, W = S_gray.shape[:2]
    S_rgb = cv2.cvtColor(S_gray, cv2.COLOR_GRAY2BGR)
    S_lb, ratio, pad = letterbox_any(S_rgb, new_shape=YOLO_SIZE)

    macro_res = get_model_macrofagos().predict(source=S_lb, imgsz=YOLO_SIZE, conf=conf_macro, verbose=False)[0]
    para_res  = get_model_parasitos().predict(source=S_lb,  imgsz=YOLO_SIZE, conf=conf_para,  verbose=False)[0]

    boxes_macro = getattr(macro_res, "boxes", None)
    boxes_para  = getattr(para_res,  "boxes", None)

    annot = S_rgb.copy()
    det_rows = []
    ok_macro = touch_macro = 0
    invalid_macro_boxes = []

    if boxes_macro is not None and len(boxes_macro) > 0:
        xyxy = unletterbox_boxes(boxes_macro.xyxy.cpu().numpy(), ratio, pad)
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 <= x1 or y2 <= y1: continue

            touches = (x1 <= TOUCH_MARGIN_PX or y1 <= TOUCH_MARGIN_PX or
                      x2 >= W-1-TOUCH_MARGIN_PX or y2 >= H-1-TOUCH_MARGIN_PX)
            if not touches and edge_mask.size > 0:
                roi = edge_mask[y1:y2, x1:x2]
                touches = roi.any()

            color = (0, 0, 255) if touches else (0, 255, 0)
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)

            if touches:
                touch_macro += 1
                invalid_macro_boxes.append([x1, y1, x2, y2])
            else:
                ok_macro += 1
            det_rows.append([x1, y1, x2, y2, 0.0, touches, "macrofago"])

    parasites_in_valid = parasites_total = 0
    if boxes_para is not None and len(boxes_para) > 0:
        xyxy = unletterbox_boxes(boxes_para.xyxy.cpu().numpy(), ratio, pad)
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            if x2 <= x1 or y2 <= y1: continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            inside_invalid = any(mx1 <= cx <= mx2 and my1 <= cy <= my2
                                 for mx1, my1, mx2, my2 in invalid_macro_boxes)

            color = (0, 0, 255) if inside_invalid else (0, 255, 0)
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)

            if not inside_invalid:
                parasites_in_valid += 1
            parasites_total += 1
            det_rows.append([x1, y1, x2, y2, 0.0, inside_invalid, "parasito"])

    total_macro = ok_macro + touch_macro
    annot = annot.astype(np.uint8)
    return annot, det_rows, total_macro, ok_macro, touch_macro, parasites_total, parasites_in_valid

# ============================================================
# ===================== UI COMPLETA ==========================
# ============================================================
tab_inicio, tab_analizar = st.tabs(["¿Cómo funciona este detector?", "Analizar"])

with tab_inicio:
    st.header("Instrucciones para usar el Detector de Leishmania")

    st.write("""
**¡Bienvenido al Detector de Leishmania!**
Esta herramienta permite analizar imágenes histológicas para la detección y el conteo automático de macrófagos y amastigotes en muestras de infección in vitro, en el marco de una etapa de validación externa del software.

⚠️ Importante: algunas funciones se encuentran en desarrollo y no deben utilizarse durante esta instancia de validación. Lea atentamente el set de instrucciones adicionales para conocer más.

## INSTRUCCIONES DE USO

### 0. Toma de imágenes
- Antes de iniciar el proceso, es imperativo contar con el conjunto de datos (dataset) definitivamente dispuesto.
  Esto implica que las imágenes deben haber sido previamente capturadas usando
  la **cámara del microscopio** o la **cámara de un teléfono móvil**.

### 1. Configuración inicial
- Ingresa un **ID del dataset** (texto único).
- Ingresa la **concentración de droga** (en μM).
- Subí tus imágenes.
  Se recomienda subir **entre 15 y 20 imágenes** (JPG, PNG o TIF) para asegurar llegar a
  **200 macrófagos válidos**.
**WARNING:** Si no alcanzás 200 macrófagos válidos, **no se generará el punto** en el gráfico.

---

### 2. Modos de imagen

## 📱 Si las imágenes se obtuvieron con un **teléfono móvil**
Normalmente tendrán un **formato circular**.

Hay dos opciones:

### ✔ **A) CON RECORTE**
- Recorte **rectangular** manual con sliders X/Y.
- **IMPORTANTE:** Si recortás, **NO** podrás usar máscara de correcciones.
- El software detecta automáticamente el **borde rectangular** para excluir macrófagos inválidos.

NOTA: UNA MÁSCARA ES UNA IMAGEN CON FONDO BLANCO DONDE SOLO SE VEN LAS IMPERFECCIONES DE LA CÁMARA. EL SOFTWARE MARCA LA UBICACIÔN
DE ESTAS MISMAS Y LAS APLICA SOBRE LAS IMAGENES A ANALIZAR ELIMINADOLAS LO MAYOR POSIBLE

### ✔ **B) SIN RECORTE**
- Se aplica el pipeline **circular** con fondo negro.
- Permite usar **máscara de imperfecciones** si querés.
- El software detecta automáticamente el **borde circular** para excluir macrófagos inválidos.

---

## 🔬 Si las imágenes se obtuvieron con **microscopio**
Siempre son **rectangulares**.

Solo hay que elegir si querés:

### ✔ **A) CON MÁSCARA**
- Se carga una máscara de imperfecciones.
- Se aplica automáticamente.
- Se usa borde rectangular.

### ✔ **B) SIN MÁSCARA**
- Se procesa directamente.
- Borde rectangular automático.

**IMPORTANTE:¡ESTA OPCIÓN AÚN ESTA EN DESAROLLO! LOS RESULTADOS NO SON DE CONFIANZA TODAVÍA**

---

### 3. Procesamiento
- Pulsá **“Procesar todas las imágenes”**.
- Verás:
  - Macrófagos **verdes** → válidos
  - Macrófagos **rojos** → tocan el borde, descartados
  - Parásitos **verdes** → válidos
  - Parásitos **rojos** → dentro de macrófagos inválidos

---

### 4. Agregar al gráfico
- Cuando llegues a 200 macrófagos válidos, tocá **“Agregar al gráfico”**.
- Se guarda un punto:
  **X:** concentración
  **Y:** parásitos por 200 macrófagos.

---

### 5. Nuevo dataset
- Podés borrar el dataset actual desde
  **“Borrar dataset actual y subir nuevas imágenes"**.

---

### Notas importantes
- **Verde = válido**
- **Rojo = descartado**
- **POR FAVOR ESPERAR** a que termine completamente la acción que solicitaste (procesar imágenes, descargar ZIP, agregar punto al gráfico, etc.) antes de realizar otra acción nueva.
- Mientras la app está procesando, verás en la **esquina superior derecha** el ícono de **un hombre en bicicleta**.
  Cuando ese ícono desaparece, significa que el proceso terminó y ya podés continuar. Esto evita errores y asegura que todos los resultados sean correctos.
- Todos los resultados se guardan en `/content/salidas`.
""")

with tab_analizar:
    st.header("Analizar imágenes")

    # -------------- Reset y estado ----------------
    def reset_dataset():
        keys = ['uploaded_images','image_modes','crop_params','use_crop','image_data','processed_images',
                'total_ok_macrofagos','total_parasites','white_mask_cell','white_mask_micro',
                'use_white_mask_cell','use_white_mask_micro','use_global_crop_cell','image_source',
                'global_crop_w','global_crop_h']
        for k in keys: st.session_state.pop(k, None)
        st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
        st.session_state.dataset_id = ""
        st.session_state.drug_concentration = 0.0

    def clear_global_history():
        st.session_state.historial_df = pd.DataFrame(columns=["dataset_id","drug_concentration","parasitos_por_200",
                                                             "macrofagos_totales","parasitos_totales","fecha","hora"])
        st.session_state.grafico_data_global = []
        for p in [HISTORIAL_CSV, GRAFICO_DATA_GLOBAL]:
            if p.exists(): p.unlink()
        reset_dataset()

    defaults = {
        'image_data': [], 'dataset_id': None, 'drug_concentration': None,
        'uploaded_images': {}, 'image_modes': {}, 'crop_params': {}, 'use_crop': {},
        'total_ok_macrofagos': 0, 'total_parasites': 0, 'processed_images': {},
        'use_white_mask_cell': False, 'use_white_mask_micro': False,
        'white_mask_cell': None, 'white_mask_micro': None,
        'use_global_crop_cell': True, 'image_source': "Dispositivo móvil",
        'global_crop_w': DEFAULT_RECT_W, 'global_crop_h': DEFAULT_RECT_H,
        'uploader_key': 0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    if 'grafico_data_global' not in st.session_state:
        st.session_state.grafico_data_global = json.load(open(GRAFICO_DATA_GLOBAL)) if GRAFICO_DATA_GLOBAL.exists() else []
    if 'historial_df' not in st.session_state:
        st.session_state.historial_df = pd.read_csv(HISTORIAL_CSV) if HISTORIAL_CSV.exists() else pd.DataFrame(columns=[
            "dataset_id","drug_concentration","parasitos_por_200","macrofagos_totales","parasitos_totales","fecha","hora"])

    # ----- Cabecera -----
    col1, col2 = st.columns([3,1])
    with col1:
        st.write("Seguí los pasos: ID → concentración → subir imágenes → configurar → procesar → agregar punto.")
    with col2:
        if st.button("Borrar HISTORIAL GLOBAL", type="secondary"):
            clear_global_history()
            st.success("Historial global borrado.")
            st.rerun()

    # ----- ID y concentración -----
    dataset_id = st.text_input("ID del dataset", value=st.session_state.dataset_id or "")
    drug_concentration = st.number_input("Concentración (μM)", value=float(st.session_state.drug_concentration or 0.0))

    if dataset_id:
        st.session_state.dataset_id = dataset_id
        st.session_state.drug_concentration = drug_concentration

    if st.button("Borrar dataset actual y subir nuevas imágenes", type="secondary"):
        reset_dataset()
        st.success("Dataset borrado. Podés cargar uno nuevo.")
        st.rerun()

    # ----- Subir imágenes -----
    uploaded_files = st.file_uploader("Sube imágenes", type=["jpg","jpeg","png","tif"], accept_multiple_files=True,
                                      key=f"uploader_{st.session_state.uploader_key}")

    if uploaded_files:
        st.session_state.uploaded_images = {f.name: f.getvalue() for f in uploaded_files}
        st.success("¡Imágenes cargadas!")

        st.session_state.image_source = st.radio("Origen de las imágenes",
            ("Cámara del microscopio", "Dispositivo móvil"),
            index=1 if st.session_state.image_source=="Dispositivo móvil" else 0)

        first_img = imread_robusto(next(iter(st.session_state.uploaded_images.values())))
        H0, W0 = first_img.shape[:2]

        if st.session_state.image_source == "Dispositivo móvil":
            recorte_opcion = st.radio("¿Querés recortar rectangularmente todas las imágenes?",
                                      ("Sí, recortar", "No, mantener circular"), index=0 if st.session_state.use_global_crop_cell else 1)
            st.session_state.use_global_crop_cell = (recorte_opcion == "Sí, recortar")

            if st.session_state.use_global_crop_cell:
                st.session_state.global_crop_w = st.slider("Ancho del rectángulo (px)", 100, W0, min(st.session_state.global_crop_w, W0))
                st.session_state.global_crop_h = st.slider("Alto del rectángulo (px)", 100, H0, min(st.session_state.global_crop_h, H0))

        st.subheader("Configuración por imagen")
        for name, data in st.session_state.uploaded_images.items():
            img = imread_robusto(data)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            colA, colB = st.columns(2)
            with colA:
                st.image(img_rgb, width=340, caption=f"Original: {name}")
            with colB:
                mode = "Celular" if st.session_state.image_source == "Dispositivo móvil" else "Microscopio"
                st.session_state.image_modes[name] = mode
                st.markdown(f"**Modo:** {mode}")

                if mode == "Celular" and st.session_state.use_global_crop_cell:
                    rect_w = min(st.session_state.global_crop_w, w)
                    rect_h = min(st.session_state.global_crop_h, h)
                    if name not in st.session_state.crop_params:
                        st.session_state.crop_params[name] = {"x": (w-rect_w)//2, "y": (h-rect_h)//2, "w": rect_w, "h": rect_h}
                    p = st.session_state.crop_params[name]
                    p["w"], p["h"] = rect_w, rect_h

                    max_x = max(0, w - rect_w)
                    max_y = max(0, h - rect_h)
                    
                    if max_x == 0 or max_y == 0:
                        st.warning("No se puede ajustar el recorte porque el tamaño de la imagen o del rectángulo no es válido.")
                        continue
                    
                    x = st.slider("Posición X", 0, max_x, p["x"], key=f"x_{name}")
                    y = st.slider("Posición Y", 0, max_y, p["y"], key=f"y_{name}")

                    st.session_state.crop_params[name]["x"] = x
                    st.session_state.crop_params[name]["y"] = y
                    st.session_state.use_crop[name] = True

                    prev = img.copy()
                    cv2.rectangle(prev, (x,y), (x+rect_w,y+rect_h), (0,255,0), 4)
                    st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), caption="Recorte (verde = válido)", width=340)
                else:
                    st.session_state.use_crop[name] = False
                    if mode == "Celular":
                        st.info("Se usará pipeline circular (fondo negro + borde automático)")

        # ----- Máscaras -----
        st.subheader("Corrección de imperfecciones (hoja blanca)")
        if st.session_state.image_source == "Dispositivo móvil":
            usar = st.radio("¿Usar hoja blanca para celular?", ("No", "Sí"), index=1 if st.session_state.use_white_mask_cell else 0)
            st.session_state.use_white_mask_cell = (usar == "Sí")
            if st.session_state.use_white_mask_cell and st.session_state.white_mask_cell is None:
                f = st.file_uploader("Subí hoja blanca (celular)", type=["jpg","png","tif"], key="wb_cell")
                if f:
                    st.session_state.white_mask_cell = build_mask_white_auto(imread_robusto(f.getvalue()))
                    st.success("Hoja blanca celular cargada")
        else:
            usar = st.radio("¿Usar hoja blanca para microscopio?", ("No", "Sí"), index=1 if st.session_state.use_white_mask_micro else 0)
            st.session_state.use_white_mask_micro = (usar == "Sí")
            if st.session_state.use_white_mask_micro and st.session_state.white_mask_micro is None:
                f = st.file_uploader("Subí hoja blanca (microscopio)", type=["jpg","png","tif"], key="wb_micro")
                if f:
                    st.session_state.white_mask_micro = build_mask_white_auto(imread_robusto(f.getvalue()))
                    st.success("Hoja blanca microscopio cargada")

        # ----- PROCESAR -----
        if st.button("Procesar todas las imágenes", type="primary"):
            out_dir = OUTPUT_DIR / (st.session_state.dataset_id or "SIN_ID")
            out_dir.mkdir(parents=True, exist_ok=True)

            prog = st.progress(0)
            status = st.empty()
            placeholder = st.empty()
            nuevas = 0

            for i, (name, data) in enumerate(st.session_state.uploaded_images.items()):
                if name in st.session_state.processed_images: continue
                status.write(f"Procesando {name}...")
                img_bgr = imread_robusto(data)
                if img_bgr is None or img_bgr.size == 0:
                    st.error(f"La imagen {name} no se pudo leer. Puede estar dañada o en formato no compatible.")
                    continue

                mode = st.session_state.image_modes[name]

                # Recorte o circular
                if mode == "Celular" and st.session_state.use_global_crop_cell and st.session_state.use_crop.get(name):
                    p = st.session_state.crop_params[name]
                    img_bgr = img_bgr[p["y"]:p["y"]+p["h"], p["x"]:p["x"]+p["w"]]
                    if img_bgr.size == 0:
                        st.error(f"El recorte de {name} quedó vacío. Ajustá los sliders.")
                        continue

                    edge_mask = make_rect_edge_mask(img_bgr.shape[0], img_bgr.shape[1])
                elif mode == "Celular":
                    img_bg, _, _, edge_mask, _, _ = circular_fondo_negro_y_borde(img_bgr)
                    if img_bg is None or img_bg.size == 0:
                        st.error(f"La detección circular falló para {name}. La imagen es demasiado pequeña o no es circular.")
                        continue

                    img_bgr = img_bg.copy()
                else:
                    edge_mask = make_rect_edge_mask(img_bgr.shape[0], img_bgr.shape[1])

                # Máscara
                white_mask = None
                if mode == "Celular" and st.session_state.use_white_mask_cell and st.session_state.white_mask_cell is not None:
                    white_mask = st.session_state.white_mask_cell
                elif mode == "Microscopio" and st.session_state.use_white_mask_micro and st.session_state.white_mask_micro is not None:
                    white_mask = st.session_state.white_mask_micro

                S = bgr_to_hsi_s_channel(img_bgr)
                S = apply_mask_on_S(S, white_mask)

                det_vis, det_rows, total_m, ok_m, touch_m, para_t, para_v = detect_on_S_with_border(
                    S, edge_mask, CONF_MACROFAGOS, CONF_PARASITOS)

                st.session_state.total_ok_macrofagos += ok_m
                st.session_state.total_parasites += para_v
                st.session_state.processed_images[name] = cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB)

                if det_vis is None or not isinstance(det_vis, np.ndarray) or det_vis.size < 10:
                    st.error(f"Ocurrió un error procesando {name}. Resultado inválido.")
                    continue
                    
                det_vis = det_vis.astype(np.uint8)
                placeholder.image(det_vis, caption=name)

                base = Path(name).stem
                cv2.imwrite(str(out_dir/f"{base}_det.jpg"), det_vis)
                pd.DataFrame(det_rows, columns=["x1","y1","x2","y2","conf","invalid","tipo"]).to_csv(out_dir/f"{base}_det.csv", index=False)

                tasa = round(para_v/ok_m*200, 2) if ok_m>0 else 0
                st.session_state.image_data.append({
                    "dataset_id": st.session_state.dataset_id, "image_name": name,
                    "ok_macrofagos": ok_m, "parasites_validos": para_v, "total_macrofagos": total_m,
                    "macrofagos_invalidos": touch_m, "parasitos_totales": para_t, "tasa_infeccion": tasa
                })

                nuevas += 1
                prog.progress((i+1)/len(st.session_state.uploaded_images))
                gc.collect()

            status.empty()
            placeholder.empty()
            if nuevas == 0:
                st.warning("Todas las imágenes ya estaban procesadas")
            else:
                st.success(f"¡{nuevas} imágenes procesadas!")
                st.balloons()

            st.write(f"**TOTAL:** {st.session_state.total_ok_macrofagos} macrófagos válidos | {st.session_state.total_parasites} parásitos")
            if st.session_state.total_ok_macrofagos >= 200:
                st.success("¡Ya podés agregar el punto al gráfico!")

    # ----- Resultados -----
    if st.session_state.processed_images:
        st.subheader("Resultados finales")
        cols = st.columns(3)
        for i, (name, img) in enumerate(st.session_state.processed_images.items()):
            with cols[i%3]:
                st.image(img, caption=name, width=380)

    if st.session_state.image_data:
        df = pd.DataFrame(st.session_state.image_data)
        st.dataframe(df, use_container_width=True)
        csv_path = OUTPUT_DIR / f"{st.session_state.dataset_id or 'SIN_ID'}_resumen.csv"
        df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button("Descargar resumen CSV", f, file_name=csv_path.name)

    # ----- Agregar punto -----
    if st.button("Agregar al gráfico", type="secondary"):
        if st.session_state.total_ok_macrofagos >= 200:
            valor = round(st.session_state.total_parasites / st.session_state.total_ok_macrofagos * 200, 2)
            st.session_state.grafico_data_global.append({"concentracion": st.session_state.drug_concentration,
                                                        "parasitos_por_200": valor})
            nueva_fila = pd.DataFrame([{
                "dataset_id": st.session_state.dataset_id,
                "drug_concentration": st.session_state.drug_concentration,
                "parasitos_por_200": valor,
                "macrofagos_totales": st.session_state.total_ok_macrofagos,
                "parasitos_totales": st.session_state.total_parasites,
                "fecha": time.strftime("%Y-%m-%d"),
                "hora": time.strftime("%H:%M:%S")
            }])
            st.session_state.historial_df = pd.concat([st.session_state.historial_df, nueva_fila], ignore_index=True)
            st.session_state.historial_df.to_csv(HISTORIAL_CSV, index=False)
            with open(GRAFICO_DATA_GLOBAL, "w") as f:
                json.dump(st.session_state.grafico_data_global, f)

            fig, ax = plt.subplots(figsize=(10,6))
            dfg = pd.DataFrame(st.session_state.grafico_data_global)
            ax.scatter(dfg["concentracion"], dfg["parasitos_por_200"], color="blue", s=100, edgecolors="black")
            for _, r in dfg.iterrows():
                ax.text(r["concentracion"], r["parasitos_por_200"], f" {r['parasitos_por_200']:.1f}", fontsize=9)
            ax.set_xlabel("Concentración (μM)"); ax.set_ylabel("Parásitos / 200 macrófagos")
            ax.set_title("Curva dosis-respuesta"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.success(f"¡Punto agregado! → {valor} parásitos/200 macrófagos")
            reset_dataset()
        else:
            st.warning(f"Faltan {200 - st.session_state.total_ok_macrofagos} macrófagos válidos")

# ==================== ZIP CORREGIDO ====================
    if st.button("Descargar ZIP del dataset actual", type="primary"):
        if not st.session_state.dataset_id:
            st.error("Ingresa un ID primero")
            st.stop()

        dataset_id = st.session_state.dataset_id.strip()
        base_salidas = BASE_DIR / "salidas"

        # 🔧 Primero aseguramos que exista
        base_salidas.mkdir(parents=True, exist_ok=True)

        # 🔍 Ahora sí podemos iterarla
        todas_las_fechas = [p for p in base_salidas.iterdir() if p.is_dir()]

        carpetas_encontradas = [
            fecha / dataset_id
            for fecha in todas_las_fechas
            if (fecha / dataset_id).exists()
        ]


        if not carpetas_encontradas:
            st.error(f"No se encontró ningún dataset con ID: **{dataset_id}**")
            st.info("Posibles causas:\n- No procesaste imágenes con ese ID aún\n- Cambiaste el ID después de procesar\n- Borraste la carpeta manualmente")
            st.stop()

        dataset_path = max(carpetas_encontradas, key=lambda x: x.stat().st_mtime)
        zip_path = BASE_DIR / f"{dataset_id}_COMPLETO.zip"

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
                for archivo in dataset_path.rglob("*"):
                    if archivo.is_file():
                        z.write(archivo, arcname=archivo.relative_to(dataset_path.parent))


            tamano_mb = zip_path.stat().st_size / (1024*1024)
            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"DESCARGAR ZIP COMPLETO — {dataset_id} ({tamano_mb:.1f} MB)",
                    data=f,
                    file_name=zip_path.name,
                    mime="application/zip",
                    type="secondary"
                )

            st.success(f"¡ZIP listo! {len(list(dataset_path.rglob('*')))} archivos comprimidos")
            st.balloons()

        except Exception as e:
            st.error(f"Error creando ZIP: {e}")

    # ----- Sidebar -----
    st.sidebar.header("Historial completo")
    if not st.session_state.historial_df.empty:
        st.sidebar.dataframe(st.session_state.historial_df, use_container_width=True)
        st.sidebar.download_button("Descargar HISTORIAL", st.session_state.historial_df.to_csv(index=False).encode(),
                                   "HISTORIAL_TODOS_DATASETS.csv", "text/csv")
    else:
        st.sidebar.info("Aún no hay datos")

    if st.session_state.grafico_data_global:
        st.sidebar.subheader("Curva dosis-respuesta")
        dfg = pd.DataFrame(st.session_state.grafico_data_global)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(dfg["concentracion"], dfg["parasitos_por_200"], color="red", s=80)
        ax2.set_xlabel("μM"); ax2.set_ylabel("Parásitos/200"); ax2.grid(True, alpha=0.3)
        st.sidebar.pyplot(fig2)
