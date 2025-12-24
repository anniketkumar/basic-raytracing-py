# English

## Ray Tracing Demo

A simple 3D ray tracing renderer built with Python and Pygame.

### Features

- Forward ray tracing with sphere primitives
- Multiple light sources with colored lighting
- Realistic color mixing (light color × object color)
- Shadow casting
- Interactive camera controls
- Diffuse lighting model

### Requirements

- Python 3.x
- Pygame

### Installation

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install pygame
```

### Usage

```bash
python app.py
```

### Controls

**Camera Movement:**
- `W` / `S` - Move forward/backward (Z-axis)
- `A` / `D` - Move left/right (X-axis)
- `Q` / `E` - Move down/up (Y-axis)

**Camera Rotation:**
- `↑` / `↓` - Rotate view up/down
- `←` / `→` - Rotate view left/right

### Color Mixing

The renderer implements physically-based color mixing where the final color is the product of light color and object color:

- White object (255,255,255) + Blue light (0,0,255) → Blue
- Yellow object (255,255,0) + Blue light (0,0,255) → Green
- Red object (255,0,0) + Blue light (0,0,255) → Black (no reflection)

### Project Structure

```
├── app.py              # Main application entry point
├── core/
│   ├── camera.py       # Camera with perspective projection
│   ├── light.py        # Light source
│   ├── ray.py          # Ray class for ray-object intersection
│   ├── sphere.py       # Sphere primitive with intersection
│   ├── vec2.py         # 2D vector math (for the 2d testing purposes)
│   └── vec3.py         # 3D vector math
└── ui/
    └── viewer.py       # Main renderer and UI loop
```

# Türkçe 

## Işın İzleme Demo

Python ve Pygame ile hazırlanmış basit bir 3B ışın izleme (ray tracing) render motoru.

### Özellikler

* Küre (sphere) primitifleri ile ileri ışın izleme (forward ray tracing)
* Renkli ışık kaynakları ile aydınlatma
* Gerçekçi renk karışımı (ışık rengi × nesne rengi)
* Gölge oluşturma
* Etkileşimli kamera kontrolleri
* Diffuse (yaygın) aydınlatma modeli

### Gereksinimler

* Python 3.x
* Pygame

### Kurulum

```bash
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install pygame
```

### Kullanım

```bash
python app.py
```

### Kontroller

**Kamera Hareketi:**

* `W` / `S` - İleri/geri hareket (Z ekseni)
* `A` / `D` - Sol/sağ hareket (X ekseni)
* `Q` / `E` - Aşağı/yukarı hareket (Y ekseni)

**Kamera Dönüşü:**

* `↑` / `↓` - Yukarı/aşağı bakış
* `←` / `→` - Sol/sağ bakış

### Renk Karışımı

Renderer, ışık rengi ile nesne renginin çarpımıyla fiziksel olarak doğru renk karışımı uygular:

* Beyaz nesne (255,255,255) + Mavi ışık (0,0,255) → Mavi
* Sarı nesne (255,255,0) + Mavi ışık (0,0,255) → Yeşil
* Kırmızı nesne (255,0,0) + Mavi ışık (0,0,255) → Siyah (yansıma yok)

### Proje Yapısı

```
├── app.py              # Ana uygulama giriş noktası
├── core/
│   ├── camera.py       # Perspektif projeksiyonlu kamera
│   ├── light.py        # Işık kaynağı
│   ├── ray.py          # Işın Sınıfı
│   ├── sphere.py       # Küre primitifi ve kesişim hesapları
│   ├── vec2.py         # 2B vektör matematiği (test amacıyla yazıldı.)
│   └── vec3.py         # 3B vektör matematiği
└── ui/
    └── viewer.py       # Render motoru ve UI döngüsü
```
