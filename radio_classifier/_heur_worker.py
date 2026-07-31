"""
Worker standalone para ProcessPoolExecutor.
Debe vivir en un .py separado (no en una celda de Jupyter) para
poder ser serializado (pickle) en Windows con spawn context.
"""

def extract_and_classify(wav_path: str):
    """
    Carga un WAV de forma ultra-rápida con soundfile, extrae métricas
    espectrales livianas con librosa (sin HPSS ni Beat Tracking) y aplica heurísticas.

    Returns
    -------
    (label, confianza, fuente)
        label     : 'musica' | None
        confianza : float  0-1
        fuente    : 'heuristica' | 'error'
    """
    try:
        import numpy as np
        import soundfile as sf
        import librosa

        # Carga ultra-rápida C con soundfile en lugar de librosa.load
        y, sr = sf.read(wav_path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)

        if len(y) < sr:               # clip demasiado corto (< 1 seg)
            return None, 0.5, 'heuristica'

        # ── Features ultra-rápidas (sin HPSS ni Beat Track) ─────────────
        # 1. Flatness Espectral (mide cuán armónico vs ruidoso/hablado es el sonido)
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # 2. Zero Crossing Rate (tasa de cruces por cero)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        # 3. RMS (Energía)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        
        # 4. Centroide Espectral (centro de masa de frecuencias)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        
        # 5. Chroma STFT (distribución de clases de tono musical)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_std = float(np.mean(np.std(chroma, axis=1)))

        # ── Heurísticas de Clasificación Rápida ─────────────────────────
        score = 0.0

        # Espectro tonal/armónico característico de música (baja flatness)
        if flatness < 0.03:
            score += 0.35
        elif flatness < 0.08:
            score += 0.20

        # Riqueza tonal en cromas
        if chroma_std > 0.18:
            score += 0.30
        elif chroma_std > 0.12:
            score += 0.15

        # Centroide espectral en rango musical habitual (1200 Hz - 3800 Hz)
        if 1200 < centroid < 3800:
            score += 0.15

        # Nivel de energía constante típico de música (RMS)
        if rms > 0.04:
            score += 0.10

        # Tasa de cruces por cero moderada (voz sibilante / ruido suele tener ZCR muy alto)
        if zcr < 0.12:
            score += 0.10

        if score >= 0.50:
            return 'musica', round(min(score, 0.95), 2), 'heuristica'
        return None, round(1.0 - score, 2), 'heuristica'

    except Exception:
        return None, 0.0, 'error'
