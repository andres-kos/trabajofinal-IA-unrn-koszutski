"""
Worker standalone para ProcessPoolExecutor en 01_segmentacion.ipynb.
Debe vivir en un archivo .py separado para ser serializable (pickle) en Windows con spawn context.
"""

import os
import io
import json
import hashlib
import subprocess
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def get_file_id(path_str: str) -> str:
    return hashlib.md5(path_str.encode()).hexdigest()[:10]


def get_duration_ffprobe(mp3_path: str, ffmpeg_bin: str) -> float:
    ffprobe_exe = str(Path(ffmpeg_bin) / 'ffprobe.exe')
    cmd = [
        ffprobe_exe,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        mp3_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'ffprobe error: {r.stderr[:300]}')
    info = json.loads(r.stdout)
    return float(info['format']['duration'])


def extract_chunk_ffmpeg(mp3_path: str, start_s: float, duration_s: float, ffmpeg_bin: str, sample_rate: int) -> AudioSegment:
    ffmpeg_exe = str(Path(ffmpeg_bin) / 'ffmpeg.exe')
    cmd = [
        ffmpeg_exe,
        '-y',
        '-ss', f'{start_s:.3f}',
        '-t',  f'{duration_s:.3f}',
        '-i',  mp3_path,
        '-f',        'wav',
        '-ar',       str(sample_rate),
        '-ac',       '1',
        '-acodec',   'pcm_s16le',
        'pipe:1',
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'ffmpeg chunk error: {r.stderr[:300]}')

    AudioSegment.converter = ffmpeg_exe
    return AudioSegment.from_wav(io.BytesIO(r.stdout))


def build_segments_from_chunk(nonsilent_ranges, min_ms, max_ms, total_chunk_ms, chunk_step_ms, is_last_chunk):
    segments  = []
    seg_start = 0

    for sound_start, sound_end in nonsilent_ranges:
        if not is_last_chunk and sound_start >= chunk_step_ms:
            break

        while sound_end - seg_start > max_ms:
            seg_end = seg_start + max_ms
            if seg_end - seg_start >= min_ms:
                segments.append((seg_start, seg_end))
            seg_start = seg_end

        if sound_start - seg_start >= min_ms:
            segments.append((seg_start, sound_start))
            seg_start = sound_end

    end_ms = total_chunk_ms if is_last_chunk else chunk_step_ms
    if end_ms - seg_start >= min_ms and (is_last_chunk or seg_start < chunk_step_ms):
        segments.append((seg_start, end_ms))

    return segments


def procesar_mp3_worker(params: dict):
    """
    Worker principal ejecutado por ProcessPoolExecutor en un proceso independiente.
    """
    mp3_path = params['mp3_path']
    temp_dir = params['temp_dir']
    ffmpeg_bin = params['ffmpeg_bin']
    min_segment_ms = params['min_segment_ms']
    max_segment_ms = params['max_segment_ms']
    min_silence_ms = params['min_silence_ms']
    silence_db_diff = params['silence_db_diff']
    sample_rate = params['sample_rate']
    chunk_s = params['chunk_s']
    overlap_s = params['overlap_s']

    try:
        total_s       = get_duration_ffprobe(mp3_path, ffmpeg_bin)
        file_id       = get_file_id(mp3_path)
        local_records = []
        seg_counter   = 0
        pos_s         = 0.0

        while pos_s < total_s:
            is_last     = (pos_s + chunk_s) >= total_s
            extract_end = min(pos_s + chunk_s + overlap_s, total_s)
            extract_dur = extract_end - pos_s

            if extract_dur < (min_segment_ms / 1000):
                break

            chunk = extract_chunk_ffmpeg(mp3_path, pos_s, extract_dur, ffmpeg_bin, sample_rate)

            if chunk.dBFS == float('-inf'):
                del chunk
                pos_s += chunk_s
                continue

            silence_thresh = chunk.dBFS - silence_db_diff
            nonsilent = detect_nonsilent(
                chunk,
                min_silence_len=min_silence_ms,
                silence_thresh=silence_thresh,
                seek_step=50
            )

            chunk_step_ms = int(chunk_s * 1000)
            segs = build_segments_from_chunk(
                nonsilent,
                min_segment_ms, max_segment_ms,
                len(chunk), chunk_step_ms, is_last
            )

            offset_ms = int(pos_s * 1000)
            for start_ms, end_ms in segs:
                seg_id   = f'{file_id}_{seg_counter:04d}'
                seg_path = os.path.join(temp_dir, f'{seg_id}.wav')
                (chunk[start_ms:end_ms]
                 .set_channels(1)
                 .set_frame_rate(sample_rate)
                 .export(seg_path, format='wav'))
                local_records.append({
                    'segment_id'    : seg_id,
                    'archivo_origen': str(mp3_path),
                    'inicio_ms'     : offset_ms + start_ms,
                    'fin_ms'        : offset_ms + end_ms,
                    'duracion_s'    : round((end_ms - start_ms) / 1000, 2),
                    'ruta_wav'      : seg_path,
                    'label'         : None,
                    'confianza'     : None,
                    'label_fuente'  : None,
                    'revisado'      : False,
                })
                seg_counter += 1

            del chunk
            pos_s += chunk_s

        if not local_records:
            return [], {'archivo': str(mp3_path), 'error': 'Sin segmentos'}
        return local_records, None

    except Exception as e:
        return [], {'archivo': str(mp3_path), 'error': str(e)}
