"""
feature_engineering.py
=======================
Beyond-ETRI 수면 예측 프로젝트 — 피처 엔지니어링 통합 모듈

노트북 3개에 흩어진 함수를 단일 파이프라인으로 통합.
  - PTDT_v5_Improved.ipynb        → Section 1, 2 (기본 집계 + 도메인 피처)
  - usage_status_wifi_hr_feature  → Section 3-A (hr / wifi / usage_stats)
  - gps_ambience_ble_feature      → Section 3-B (gps / ambience / ble)

Usage:
    from feature_engineering import run_feature_pipeline
    feature_df = run_feature_pipeline(raw)   # raw: FILE_MAP 로드 결과 dict
"""

import pandas as pd
import numpy as np
import warnings
from functools import reduce

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Section 0. 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def get_timestamp_col(df):
    """타임스탬프 컬럼 자동 탐지 (PTDT_v5 기준)"""
    for c in df.columns:
        if any(k in c.lower() for k in ['timestamp', 'time', 'datetime', 'ts', 'date']):
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def ensure_datetime(df, ts_col='timestamp'):
    """타임스탬프 파싱 + date/hour 컬럼 추가 (gps_ambience_ble 기준)"""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df['date'] = pd.to_datetime(df[ts_col].dt.date)
    df['hour'] = df[ts_col].dt.hour
    return df


def to_list(x):
    """중첩 컬럼 값을 list로 안전하게 변환"""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


# ── 도메인 피처용 내부 유틸 ──────────────────────────────────────────────────

# 시간대 구간 정의
DAYTIME     = (8,  21)   # 주간
PRESLEEP    = (21, 24)   # 취침 전
SLEEP_NIGHT = (0,  7)    # 야간 (수면 중)


def _parse_ts(df):
    """도메인 피처 전용 타임스탬프 파서: _dt / date / hour 컬럼 추가"""
    df = df.copy()
    ts_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['timestamp', 'time', 'ts', 'datetime']):
            ts_col = c
            break
    if ts_col is None:
        return None
    if df[ts_col].dtype in ['int64', 'float64']:
        df['_dt'] = pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
    else:
        df['_dt'] = pd.to_datetime(df[ts_col], errors='coerce')
    df['date'] = df['_dt'].dt.normalize()
    df['hour'] = df['_dt'].dt.hour
    return df


def _wmask(df, h_start, h_end):
    """시간대 boolean 마스크 (자정 넘김 구간 지원)"""
    if h_end >= h_start:
        return (df['hour'] >= h_start) & (df['hour'] < h_end)
    return (df['hour'] >= h_start) | (df['hour'] < h_end)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1. 기본 수치형 센서 집계 (PTDT_v5 기준)
# ══════════════════════════════════════════════════════════════════════════════

def extract_daily_features(df, sensor_name, subject_col='subject_id'):
    """
    수치형 컬럼을 가진 센서 → subject_id + date 기준 일별 집계 피처
    집계: mean / std / min / max / count
    """
    df = df.copy()
    ts_col = get_timestamp_col(df)
    if ts_col and ts_col in df.columns:
        df[ts_col] = (
            pd.to_datetime(df[ts_col], unit='ms', errors='coerce')
            if df[ts_col].dtype in ['int64', 'float64']
            else pd.to_datetime(df[ts_col], errors='coerce')
        )
        df['date'] = df[ts_col].dt.date
    elif 'date' not in df.columns:
        print(f'  ⚠️  [{sensor_name}] 타임스탬프 없음')
        return None

    df['date'] = pd.to_datetime(df['date'])

    exclude  = {subject_col, 'date', ts_col}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude]
    if not num_cols:
        return None

    grp   = df.groupby([subject_col, 'date'])[num_cols]
    parts = []
    for agg_name, agg_df in {
        'mean':  grp.mean(),
        'std':   grp.std().fillna(0),
        'min':   grp.min(),
        'max':   grp.max(),
        'count': grp.count()
    }.items():
        agg_df.columns = [f'{sensor_name}__{c}__{agg_name}' for c in agg_df.columns]
        parts.append(agg_df)

    result = pd.concat(parts, axis=1).reset_index()
    print(f'  ✅ [{sensor_name}]: {result.shape[1]-2}개 피처')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Section 2. 도메인 특화 피처 — 시간대별 집계 (PTDT_v5 4-B)
#   주간(08-21), 취침전(21-24), 야간(00-07) 구간별로 레이블 연관 피처 추출
#   Q1/Q2/Q3 (주관) 및 S1~S4 (객관) 레이블 각각에 타겟 피처 대응
# ══════════════════════════════════════════════════════════════════════════════

def _feat_hr(df):
    """HR: 야간 평균/분/std(Q1), 주간 최대(Q2), 취침전 트렌드(Q3), 수면 분율(S1/S4)"""
    df = _parse_ts(df)
    if df is None:
        return pd.DataFrame()
    hr_col = next((c for c in df.columns
                   if 'heart' in c.lower() or c.lower() in ['hr', 'bpm']), None)
    if hr_col is None:
        return pd.DataFrame()
    rows = []
    for (subj, date), g in df.groupby(['subject_id', 'date']):
        r     = {'subject_id': subj, 'date': date}
        day   = g[_wmask(g, *DAYTIME)][hr_col].dropna()
        pre   = g[_wmask(g, *PRESLEEP)][hr_col].dropna()
        night = g[_wmask(g, *SLEEP_NIGHT)][hr_col].dropna()
        r['hr_day_mean']          = day.mean()
        r['hr_day_max']           = day.max()
        r['hr_day_above100']      = (day > 100).mean()
        r['hr_day_std']           = day.std()
        r['hr_presleep_mean']     = pre.mean()
        try:
            n = len(pre) // 3
            r['hr_presleep_trend'] = (
                pre.iloc[-n:].mean() - pre.iloc[:n].mean()
                if n > 0 else np.nan
            )
        except Exception:
            r['hr_presleep_trend'] = np.nan
        r['hr_night_mean']        = night.mean()
        r['hr_night_std']         = night.std()
        r['hr_night_min']         = night.min()
        r['hr_night_sleep_frac']  = (night < 60).mean() if len(night) > 0 else np.nan
        if len(night) > 10:
            above = (night > 75).astype(int)
            r['hr_night_spikes']  = (above.diff().fillna(0) == 1).sum()
        else:
            r['hr_night_spikes']  = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _feat_pedo(df):
    """Pedo: 걸음수/활동분(Q2), 비활동시간(Q3), 야간 zero-step 분율(S1)"""
    df = _parse_ts(df)
    if df is None:
        return pd.DataFrame()
    steps_col = next((c for c in df.columns if 'step' in c.lower()), None)
    dist_col  = next((c for c in df.columns if 'dist' in c.lower()), None)
    cal_col   = next((c for c in df.columns if 'cal'  in c.lower()), None)
    rows = []
    for (subj, date), g in df.groupby(['subject_id', 'date']):
        r = {'subject_id': subj, 'date': date}
        if steps_col:
            day_s = g[_wmask(g, *DAYTIME)][steps_col].dropna()
            r['pedo_total_steps']     = day_s.sum()
            r['pedo_active_mins']     = (day_s > 0).sum()
            r['pedo_peak_steps']      = day_s.max()
            night_s = g[_wmask(g, *SLEEP_NIGHT)][steps_col].dropna()
            r['pedo_night_zero_frac'] = (night_s == 0).mean()
            r['pedo_sedentary_hrs']   = (day_s < 5).sum() / 60
        if cal_col:
            r['pedo_total_calories']  = g[cal_col].sum()
        if dist_col:
            r['pedo_total_distance']  = g[dist_col].sum()
        rows.append(r)
    return pd.DataFrame(rows)


def _feat_activity(df):
    """Activity: 격렬 분율(Q2), 주간 정지(Q3), 야간 이동(S2/Q1), 마지막 활동 시각(S3)"""
    df = _parse_ts(df)
    if df is None:
        return pd.DataFrame()
    act_col = next((c for c in df.columns
                    if 'activity' in c.lower() or 'type' in c.lower()), None)
    if act_col is None:
        return pd.DataFrame()
    STILL    = 0
    VIGOROUS = [4, 8]
    rows = []
    for (subj, date), g in df.groupby(['subject_id', 'date']):
        r = {'subject_id': subj, 'date': date}
        g = g.sort_values('_dt')
        day   = g[_wmask(g, *DAYTIME)][act_col].dropna()
        night = g[_wmask(g, *SLEEP_NIGHT)][act_col].dropna()
        pre   = g[_wmask(g, *PRESLEEP)].sort_values('_dt')
        r['act_vigorous_frac']           = day.isin(VIGOROUS).mean()
        r['act_walking_frac']            = (day == 2).mean()
        r['act_still_day_frac']          = (day == STILL).mean()
        r['act_still_night_frac']        = (night == STILL).mean()
        r['act_night_movement_count']    = (night != STILL).sum()
        if len(pre) > 0:
            ns = pre[pre[act_col] != STILL]
            r['act_presleep_last_active_hour'] = ns['hour'].iloc[-1] if len(ns) > 0 else 21
        rows.append(r)
    return pd.DataFrame(rows)


def _feat_screen(df):
    """Screen: 취침전 ON 횟수(Q3), 야간 OFF 분율(S1), 마지막 OFF 시각(S3), 야간 ON 횟수(S4)"""
    df = _parse_ts(df)
    if df is None:
        return pd.DataFrame()
    st_col = next((c for c in df.columns
                   if 'status' in c.lower() or 'screen' in c.lower()
                   or 'state' in c.lower()), None)
    if st_col is None:
        return pd.DataFrame()
    rows = []
    for (subj, date), g in df.groupby(['subject_id', 'date']):
        r = {'subject_id': subj, 'date': date}
        g = g.sort_values('_dt')
        pre   = g[_wmask(g, *PRESLEEP)]
        night = g[_wmask(g, *SLEEP_NIGHT)]
        if len(pre) > 0:
            on = (pre[st_col] == 1)
            r['screen_presleep_on_count'] = on.sum()
            r['screen_presleep_on_frac']  = on.mean()
            off_t = pre[pre[st_col] == 0]['_dt']
            if len(off_t) > 0:
                t = off_t.iloc[-1]
                r['screen_last_off_hour'] = t.hour + t.minute / 60.0
            else:
                r['screen_last_off_hour'] = 21.0
        if len(night) > 0:
            r['screen_night_off_frac']  = (night[st_col] == 0).mean()
            on_diff = (night[st_col] == 1).astype(int).diff().fillna(0)
            r['screen_night_on_count']  = (on_diff == 1).sum()
        r['screen_total_on_count'] = (g[st_col] == 1).sum()
        rows.append(r)
    return pd.DataFrame(rows)


def _feat_light(df, sname='mLight'):
    """Light: 야외 분율(Q2), 야간 어둠 분율(Q1/S1), 취침전 조도 감소(S3)"""
    df = _parse_ts(df)
    if df is None:
        return pd.DataFrame()
    lux = next((c for c in df.columns
                if any(k in c.lower() for k in ['lux', 'light', 'illum', 'bright'])), None)
    if lux is None:
        return pd.DataFrame()
    rows = []
    for (subj, date), g in df.groupby(['subject_id', 'date']):
        r = {'subject_id': subj, 'date': date}
        day   = g[_wmask(g, *DAYTIME)][lux].dropna()
        pre   = g[_wmask(g, *PRESLEEP)][lux].dropna()
        night = g[_wmask(g, *SLEEP_NIGHT)][lux].dropna()
        r[f'{sname}_day_mean']        = day.mean()
        r[f'{sname}_outdoor_frac']    = (day > 1000).mean()
        r[f'{sname}_night_mean']      = night.mean()
        r[f'{sname}_night_dark_frac'] = (night < 10).mean()
        if len(pre) >= 4:
            h = len(pre) // 2
            r[f'{sname}_presleep_decrease'] = pre.iloc[:h].mean() - pre.iloc[h:].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def build_domain_features(raw_dict):
    """
    도메인 시간대 피처 통합 빌더 (PTDT_v5 build_domain_features 그대로 유지)
    raw_dict 내 각 센서에 개별 추출 함수를 적용 후 outer merge
    """
    extractors = {
        'pedo'       : lambda d: _feat_pedo(d),
        'activity'   : lambda d: _feat_activity(d),
        'screen'     : lambda d: _feat_screen(d),
        'light'      : lambda d: _feat_light(d, 'mLight'),
        'w_light'    : lambda d: _feat_light(d, 'wLight'),
    }
    dfs = []
    for key, fn in extractors.items():
        if key not in raw_dict or raw_dict[key] is None:
            continue
        print(f'  [domain] {key}...')
        try:
            fdf = fn(raw_dict[key])
            if len(fdf) > 0:
                dfs.append(fdf)
                print(f'    → {fdf.shape[1]-2}개 피처')
        except Exception as e:
            print(f'    ⚠️  {key}: {e}')
    if not dfs:
        print('  ⚠️  domain 피처 없음')
        return pd.DataFrame()
    result = reduce(
        lambda l, r: pd.merge(l, r, on=['subject_id', 'date'], how='outer'), dfs
    )
    print(f'  ✅ domain 피처 완료: {result.shape[1]-2}개')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Section 3-A. 중첩 센서 특화 함수 — HR / WiFi / UsageStats
#   (usage_status_wifi_hr_feature.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def extract_hr_features_nested(df, subject_col='subject_id'):
    """heart_rate: List[int] → 일별 기본 통계 + 야간/취침전 HR"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = pd.to_datetime(df['date'])

    df = df.explode('heart_rate')
    df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
    df = df.dropna(subset=['heart_rate'])
    df = df[df['heart_rate'].between(30, 220)]

    results = []
    for (subj, date), grp in df.groupby([subject_col, 'date']):
        hr        = grp['heart_rate']
        night     = grp[grp['hour'].between(0, 6)]['heart_rate']
        pre_sleep = grp[grp['hour'].between(21, 23)]['heart_rate']

        results.append({
            subject_col:         subj,
            'date':              date,
            'hr__mean':          hr.mean(),
            'hr__std':           hr.std(ddof=0),
            'hr__min':           hr.min(),
            'hr__max':           hr.max(),
            'hr__count':         len(hr),
            'hr__resting':       hr.quantile(0.1),
            'hr__rmssd':         float(np.sqrt(np.mean(np.diff(hr.values)**2)))
                                 if len(hr) > 1 else 0,
            'hr__night_mean':    night.mean()    if len(night) > 0 else np.nan,
            'hr__night_std':     night.std(ddof=0) if len(night) > 1 else 0,
            'hr__night_min':     night.min()     if len(night) > 0 else np.nan,
            'hr__presleep_mean': pre_sleep.mean() if len(pre_sleep) > 0 else np.nan,
        })

    result_df = pd.DataFrame(results)
    print(f'  ✅ [hr_nested]: {result_df.shape[1]-2}개 피처')
    return result_df


def estimate_sleep_from_hr(df, subject_col='subject_id',
                            min_sleep_duration_min=60):
    """
    HR이 개인 임계값(하위 35%) 아래로 sustained하게 떨어지는 구간 = 수면 구간
    → 수면 시작/종료/길이/HR 통계/각성 이벤트 8개 피처
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df = df.explode('heart_rate')
    df['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
    df = df.dropna(subset=['heart_rate', 'timestamp'])
    df = df[df['heart_rate'].between(30, 220)]
    df = df.sort_values([subject_col, 'timestamp'])

    results = []
    for subj, subj_df in df.groupby(subject_col):
        sleep_threshold = subj_df['heart_rate'].quantile(0.35)
        dates = subj_df['timestamp'].dt.date.unique()

        for date in dates:
            date         = pd.Timestamp(date)
            window_start = date - pd.Timedelta(hours=4)
            window_end   = date + pd.Timedelta(hours=12)

            window = subj_df[
                (subj_df['timestamp'] >= window_start) &
                (subj_df['timestamp'] <= window_end)
            ].copy()
            if len(window) < 10:
                continue

            window       = window.set_index('timestamp')['heart_rate']
            window_1min  = window.resample('1min').mean().interpolate()
            is_low       = window_1min < sleep_threshold

            sleep_onset = sleep_offset = None
            max_duration = 0
            in_sleep = False
            seg_start = None

            for t, low in is_low.items():
                if low and not in_sleep:
                    in_sleep  = True
                    seg_start = t
                elif not low and in_sleep:
                    duration = (t - seg_start).total_seconds() / 60
                    if duration > max_duration and duration >= min_sleep_duration_min:
                        max_duration  = duration
                        sleep_onset   = seg_start
                        sleep_offset  = t
                    in_sleep = False

            row = {subject_col: subj, 'date': date}
            if sleep_onset and sleep_offset:
                sleep_hr = window_1min[sleep_onset:sleep_offset]
                pre_sleep_hr = window_1min[
                    max(window_1min.index[0], sleep_onset - pd.Timedelta(hours=1))
                    :sleep_onset
                ]
                row.update({
                    'hr__sleep_onset_hour':  sleep_onset.hour + sleep_onset.minute / 60,
                    'hr__sleep_offset_hour': sleep_offset.hour + sleep_offset.minute / 60,
                    'hr__est_tst_min':       max_duration,
                    'hr__sleep_hr_mean':     sleep_hr.mean(),
                    'hr__sleep_hr_std':      sleep_hr.std(),
                    'hr__sleep_hr_min':      sleep_hr.min(),
                    'hr__presleep_hr_drop':  (pre_sleep_hr.mean() - sleep_hr.mean()
                                             if len(pre_sleep_hr) > 0 else np.nan),
                    'hr__arousal_count':     int((sleep_hr > sleep_threshold).sum()),
                })
            else:
                for col in ['hr__sleep_onset_hour', 'hr__sleep_offset_hour',
                            'hr__est_tst_min', 'hr__sleep_hr_mean', 'hr__sleep_hr_std',
                            'hr__sleep_hr_min', 'hr__presleep_hr_drop', 'hr__arousal_count']:
                    row[col] = np.nan
            results.append(row)

    result_df = pd.DataFrame(results)
    print(f'  ✅ [hr_sleep]: {result_df.shape[1]-2}개 피처')
    return result_df


def extract_wifi_features_nested(df, subject_col='subject_id'):
    """m_wifi: List[{bssid, rssi}] → 이동성·귀가·신호강도 피처"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = pd.to_datetime(df['date'])

    df = df.explode('m_wifi').dropna(subset=['m_wifi'])
    wifi_expanded = pd.json_normalize(df['m_wifi'].tolist())
    df = pd.concat([
        df[[subject_col, 'date', 'hour', 'timestamp']].reset_index(drop=True),
        wifi_expanded.reset_index(drop=True)
    ], axis=1)

    if 'rssi' in df.columns:
        df['rssi'] = pd.to_numeric(df['rssi'], errors='coerce')

    home_bssid = (df.groupby(subject_col)['bssid']
                    .agg(lambda x: x.value_counts().index[0])
                    .to_dict()) if 'bssid' in df.columns else {}
    if home_bssid:
        df['is_home'] = df.apply(
            lambda r: int(r['bssid'] == home_bssid.get(r[subject_col], '')), axis=1)

    results = []
    for (subj, date), grp in df.groupby([subject_col, 'date']):
        night_grp = grp[grp['hour'].between(0, 6)]
        row = {
            subject_col:               subj,
            'date':                    date,
            'wifi__scan_count':        len(grp),
            'wifi__unique_bssid':      grp['bssid'].nunique() if 'bssid' in grp else 0,
            'wifi__night_scan_count':  len(night_grp),
            'wifi__home_ratio':        grp['is_home'].mean() if 'is_home' in grp else 0,
            'wifi__home_count':        grp['is_home'].sum()  if 'is_home' in grp else 0,
        }
        if 'rssi' in grp.columns and grp['rssi'].notna().sum() > 0:
            row['wifi__rssi_mean'] = grp['rssi'].mean()
            row['wifi__rssi_std']  = grp['rssi'].std(ddof=0)
            row['wifi__rssi_home_mean'] = (
                grp[grp['is_home'] == 1]['rssi'].mean()
                if 'is_home' in grp and grp['is_home'].sum() > 0
                else np.nan
            )
        if 'is_home' in grp.columns:
            evening_home = grp[(grp['is_home'] == 1) & (grp['hour'] >= 18)]
            row['wifi__home_arrival_hour'] = (
                evening_home['hour'].min() if len(evening_home) > 0 else np.nan
            )
        results.append(row)

    result_df = pd.DataFrame(results)
    print(f'  ✅ [wifi_nested]: {result_df.shape[1]-2}개 피처')
    return result_df


def extract_wifi_bedtime_proxy(df, subject_col='subject_id'):
    """
    야간 WiFi 스캔 끊김 → 폰 내려놓은 시각 추정
    → phone_down_hour / last_night_scan_hour / first_morning_scan_hour 3개 피처
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['timestamp'].dt.hour

    results = []
    for (subj, date), grp in df.groupby([subject_col, 'date']):
        night = grp[grp['hour'].between(20, 23) | grp['hour'].between(0, 6)]
        night = night.sort_values('timestamp')
        row   = {subject_col: subj, 'date': date}

        if len(night) >= 2:
            night = night.set_index('timestamp')
            gaps  = night.index.to_series().diff().dt.total_seconds() / 60
            long_gap = gaps[gaps > 30]
            if len(long_gap) > 0:
                gap_start = long_gap.index[0] - pd.Timedelta(minutes=gaps[long_gap.index[0]])
                row['wifi__phone_down_hour'] = gap_start.hour + gap_start.minute / 60
            else:
                row['wifi__phone_down_hour'] = np.nan

            last_scan = night.index[-1]
            row['wifi__last_night_scan_hour'] = last_scan.hour + last_scan.minute / 60

            morning = grp[grp['hour'].between(5, 9)].sort_values('timestamp')
            row['wifi__first_morning_scan_hour'] = (
                morning['timestamp'].iloc[0].hour + morning['timestamp'].iloc[0].minute / 60
                if len(morning) > 0 else np.nan
            )
        else:
            row.update({
                'wifi__phone_down_hour':        np.nan,
                'wifi__last_night_scan_hour':   np.nan,
                'wifi__first_morning_scan_hour': np.nan,
            })
        results.append(row)

    result_df = pd.DataFrame(results)
    print(f'  ✅ [wifi_bedtime]: {result_df.shape[1]-2}개 피처')
    return result_df


def extract_usage_stats_features_nested(df, subject_col='subject_id'):
    """m_usage_stats: List[{app_name, total_time}] → 앱 카테고리별 사용시간 피처"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = pd.to_datetime(df['date'])

    df = df.explode('m_usage_stats').dropna(subset=['m_usage_stats'])
    usage_expanded = pd.json_normalize(df['m_usage_stats'].tolist())
    df = pd.concat([
        df[[subject_col, 'date', 'hour']].reset_index(drop=True),
        usage_expanded.reset_index(drop=True)
    ], axis=1)

    if 'total_time' in df.columns:
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce').fillna(0)
        df['total_time_min'] = df['total_time'] / 60000

    CATEGORIES = {
        'social':  ['kakao', 'instagram', 'facebook', 'twitter', 'tiktok',
                    'snapchat', 'line', 'telegram', 'whatsapp', 'naver.band'],
        'video':   ['youtube', 'netflix', 'tving', 'watcha', 'wavve', 'twitch', 'video'],
        'game':    ['game', 'com.nexon', 'com.netmarble', 'pubg', 'minecraft'],
        'browser': ['chrome', 'samsung.internet', 'firefox', 'naver', 'browser'],
        'work':    ['office', 'notion', 'slack', 'zoom', 'teams',
                    'gmail', 'outlook', 'calendar', 'docs'],
    }

    def categorize(app_name):
        if pd.isna(app_name):
            return 'other'
        name = str(app_name).lower()
        for cat, keywords in CATEGORIES.items():
            if any(k in name for k in keywords):
                return cat
        return 'other'

    if 'app_name' in df.columns:
        df['category'] = df['app_name'].apply(categorize)

    results = []
    for (subj, date), grp in df.groupby([subject_col, 'date']):
        row = {subject_col: subj, 'date': date}

        if 'total_time_min' in grp.columns:
            row['usage__total_use_min'] = grp['total_time_min'].sum()
            night_grp = grp[grp['hour'].between(0, 6)]
            row['usage__night_use_min'] = night_grp['total_time_min'].sum()

            if 'category' in grp.columns:
                for cat in CATEGORIES:
                    cat_grp = grp[grp['category'] == cat]
                    row[f'usage__{cat}_min'] = cat_grp['total_time_min'].sum()
        results.append(row)

    result_df = pd.DataFrame(results)
    print(f'  ✅ [usage_nested]: {result_df.shape[1]-2}개 피처')
    return result_df


def extract_usage_bedtime_proxy(df, subject_col='subject_id'):
    """
    마지막 앱 사용 시각 → 취침 시각 / 첫 아침 사용 → 기상 시각
    → last_use_hour / first_morning_hour / sleep_hour_count / phone_off_duration_hr 4개 피처
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['timestamp'].dt.hour

    results = []
    for (subj, date), grp in df.groupby([subject_col, 'date']):
        row = {subject_col: subj, 'date': date}

        evening = grp[grp['hour'] >= 20].sort_values('timestamp')
        row['usage__last_use_hour'] = (
            evening['timestamp'].iloc[-1].hour + evening['timestamp'].iloc[-1].minute / 60
            if len(evening) > 0 else np.nan
        )

        morning = grp[grp['hour'].between(5, 10)].sort_values('timestamp')
        row['usage__first_morning_hour'] = (
            morning['timestamp'].iloc[0].hour + morning['timestamp'].iloc[0].minute / 60
            if len(morning) > 0 else np.nan
        )

        row['usage__sleep_hour_count'] = grp['hour'].between(0, 6).sum()

        if not pd.isna(row.get('usage__last_use_hour')) and \
           not pd.isna(row.get('usage__first_morning_hour')):
            onset  = row['usage__last_use_hour']
            offset = row['usage__first_morning_hour']
            gap    = (offset - onset) if offset > onset else (24 - onset + offset)
            row['usage__phone_off_duration_hr'] = gap
        else:
            row['usage__phone_off_duration_hr'] = np.nan

        results.append(row)

    result_df = pd.DataFrame(results)
    print(f'  ✅ [usage_bedtime]: {result_df.shape[1]-2}개 피처')
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# Section 3-B. 중첩 센서 특화 함수 — GPS / Ambience / BLE
#   (gps_ambience_ble_feature.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def extract_gps_special_features(df, subject_col='subject_id'):
    """
    m_gps: List[{altitude, latitude, longitude, speed}]
    → 이동성 / 야간 이동 / 마지막 이동 시각 / 방문 장소 수 등 피처
    """
    df = ensure_datetime(df)
    rows = []

    for (subj, date), g in df.groupby([subject_col, 'date']):
        latitudes, longitudes, altitudes, speeds = [], [], [], []
        moving_times       = []
        night_speed_flags  = []

        for _, row in g.iterrows():
            ts   = row['timestamp']
            hour = row['hour']
            for item in to_list(row['m_gps']):
                if isinstance(item, dict):
                    if item.get('latitude')  is not None: latitudes.append(float(item['latitude']))
                    if item.get('longitude') is not None: longitudes.append(float(item['longitude']))
                    if item.get('altitude')  is not None: altitudes.append(float(item['altitude']))
                    if item.get('speed')     is not None:
                        spd = float(item['speed'])
                        if 0 <= spd <= 30:
                            speeds.append(spd)
                            is_moving = spd > 0.5
                            if is_moving:
                                moving_times.append(ts)
                            if 0 <= hour <= 5:
                                night_speed_flags.append(is_moving)

        latitudes  = np.array(latitudes)
        longitudes = np.array(longitudes)
        altitudes  = np.array(altitudes)
        speeds     = np.array(speeds)

        if len(moving_times) > 0:
            last_move_time       = max(moving_times)
            last_movement_hour   = last_move_time.hour + last_move_time.minute / 60
            if last_movement_hour < 6:
                last_movement_hour += 24
        else:
            last_movement_hour = 0

        feat = {
            'subject_id':                  subj,
            'date':                        date,
            'gps__record_count':           len(g),
            'gps__point_count':            len(speeds),
            'gps__speed_mean':             np.mean(speeds)  if len(speeds) else 0,
            'gps__speed_std':              np.std(speeds)   if len(speeds) else 0,
            'gps__speed_max':              np.max(speeds)   if len(speeds) else 0,
            'gps__moving_ratio':           np.mean(speeds > 0.5) if len(speeds) else 0,
            'gps__lat_std':                np.std(latitudes)  if len(latitudes) else 0,
            'gps__lon_std':                np.std(longitudes) if len(longitudes) else 0,
            'gps__altitude_mean':          np.mean(altitudes) if len(altitudes) else 0,
            'gps__altitude_std':           np.std(altitudes)  if len(altitudes) else 0,
            'gps__last_movement_hour':     last_movement_hour,
            'gps__night_moving_ratio':     np.mean(night_speed_flags) if len(night_speed_flags) else 0,
        }
        if len(latitudes) and len(longitudes):
            rough_places = set(zip(np.round(latitudes, 3), np.round(longitudes, 3)))
            feat['gps__unique_rough_place_count'] = len(rough_places)
        else:
            feat['gps__unique_rough_place_count'] = 0

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f'  ✅ [gps_special]: {result.shape[1]-2}개 피처')
    return result


def extract_ambience_special_features(df, subject_col='subject_id'):
    """
    m_ambience: List[List[str]] (이중 중첩 — [label, score] 쌍)
    → 12개 환경음 레이블별 mean_score / max_score / top1_ratio + 저녁 구간 피처
    """
    df = ensure_datetime(df)

    TARGET_LABELS = [
        'Speech', 'Music', 'Vehicle', 'Motor vehicle (road)', 'Car',
        'Outside, urban or manmade', 'Outside, rural or natural',
        'Inside, large room or hall', 'Inside, small room',
        'Animal', 'Silence', 'Noise'
    ]
    rows = []

    for (subj, date), g in df.groupby([subject_col, 'date']):
        sums       = {label: 0.0 for label in TARGET_LABELS}
        maxs       = {label: 0.0 for label in TARGET_LABELS}
        top1_counts = {label: 0   for label in TARGET_LABELS}

        total_events   = 0
        evening_events = 0
        evening_speech = evening_music = evening_vehicle = 0.0

        for _, row in g.iterrows():
            arr  = to_list(row.get('m_ambience'))
            hour = row['hour']
            scores = {}

            for item in arr:
                item = to_list(item)
                if len(item) >= 2:
                    label = str(item[0])
                    try:    score = float(item[1])
                    except: score = 0.0
                    scores[label] = score
                    if label in TARGET_LABELS:
                        sums[label] += score
                        maxs[label]  = max(maxs[label], score)

            if scores:
                total_events += 1
                top_label = max(scores, key=scores.get)
                if top_label in top1_counts:
                    top1_counts[top_label] += 1
                if 20 <= hour <= 23:
                    evening_events += 1
                    evening_speech  += scores.get('Speech', 0.0)
                    evening_music   += scores.get('Music',  0.0)
                    evening_vehicle += (scores.get('Vehicle', 0.0) +
                                        scores.get('Motor vehicle (road)', 0.0))

        feat = {
            'subject_id':                      subj,
            'date':                            date,
            'ambience__record_count':          len(g),
            'ambience__valid_event_count':     total_events,
            'ambience__evening_event_count':   evening_events,
            'ambience__evening_speech_mean':   evening_speech / evening_events if evening_events else 0,
            'ambience__evening_music_mean':    evening_music  / evening_events if evening_events else 0,
            'ambience__evening_vehicle_mean':  evening_vehicle / evening_events if evening_events else 0,
        }
        for label in TARGET_LABELS:
            clean = (label.lower()
                     .replace(' ', '_').replace(',', '')
                     .replace('(', '').replace(')', '').replace('/', '_'))
            feat[f'ambience__{clean}__mean_score']  = sums[label] / total_events if total_events else 0
            feat[f'ambience__{clean}__max_score']   = maxs[label]
            feat[f'ambience__{clean}__top1_ratio']  = top1_counts[label] / total_events if total_events else 0

        rows.append(feat)

    result = pd.DataFrame(rows).fillna(0)
    print(f'  ✅ [ambience_special]: {result.shape[1]-2}개 피처')
    return result


def extract_ble_special_features(df, subject_col='subject_id'):
    """
    m_ble: List[{address, device_class, rssi}]
    → 스캔 수 / 고유 기기 수 / RSSI 통계 / 저녁·야간 기기 수 피처
    """
    df = ensure_datetime(df)
    rows = []

    for (subj, date), g in df.groupby([subject_col, 'date']):
        scan_device_counts = []
        rssis              = []
        unique_addresses   = set()
        device_classes     = set()
        evening_counts     = []
        night_counts       = []

        for _, row in g.iterrows():
            devices = to_list(row['m_ble'])
            hour    = row['hour']
            count   = 0
            for dev in devices:
                if isinstance(dev, dict):
                    count += 1
                    addr  = dev.get('address')
                    if addr:
                        unique_addresses.add(addr)
                    dc = dev.get('device_class')
                    if dc is not None:
                        device_classes.add(str(dc))
                    rssi = dev.get('rssi')
                    if rssi is not None:
                        rssis.append(float(rssi))

            scan_device_counts.append(count)
            if 20 <= hour <= 23: evening_counts.append(count)
            if 0  <= hour <= 5:  night_counts.append(count)

        scan_device_counts = np.array(scan_device_counts)
        rssis              = np.array(rssis)

        feat = {
            'subject_id':                     subj,
            'date':                           date,
            'ble__scan_count':                len(g),
            'ble__device_count_mean':         np.mean(scan_device_counts) if len(scan_device_counts) else 0,
            'ble__device_count_std':          np.std(scan_device_counts)  if len(scan_device_counts) else 0,
            'ble__device_count_max':          np.max(scan_device_counts)  if len(scan_device_counts) else 0,
            'ble__unique_device_count':       len(unique_addresses),
            'ble__unique_device_class_count': len(device_classes),
            'ble__rssi_mean':                 np.mean(rssis)  if len(rssis) else 0,
            'ble__rssi_std':                  np.std(rssis)   if len(rssis) else 0,
            'ble__rssi_max':                  np.max(rssis)   if len(rssis) else 0,
            'ble__strong_signal_ratio':       np.mean(rssis > -60) if len(rssis) else 0,
            'ble__evening_device_count_mean': np.mean(evening_counts) if len(evening_counts) else 0,
            'ble__night_device_count_mean':   np.mean(night_counts)   if len(night_counts) else 0,
        }
        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f'  ✅ [ble_special]: {result.shape[1]-2}개 피처')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Section 4. 통합 진입점
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_pipeline(raw: dict) -> pd.DataFrame:
    """
    raw: FILE_MAP 기준으로 로드된 {sensor_name: DataFrame} dict
    returns: subject_id + date 기준 통합 feature_df

    파이프라인 순서
    ───────────────
    1. 기본 수치형 센서 일별 집계 (ac_status / light / screen / w_light / pedo)
    2. 도메인 시간대 피처 병합 (hr / pedo / activity / screen / light / w_light)
    3. 중첩 센서 HR  (nested + sleep estimation)
    4. 중첩 센서 WiFi (nested + bedtime proxy)
    5. 중첩 센서 UsageStats (nested + bedtime proxy)
    6. 중첩 센서 GPS / Ambience / BLE
    7. 전체 outer merge → feature_df 반환
    """
    daily = {}

    print('📊 [1] 기본 수치형 센서 집계')
    basic_sensors = ['ac_status', 'activity', 'light', 'screen', 'w_light', 'pedo']
    for key in basic_sensors:
        if key in raw:
            feat = extract_daily_features(raw[key], key)
            if feat is not None:
                daily[key] = feat

    print('📊 [2] 도메인 시간대 피처')
    domain_df = build_domain_features(raw)
    if len(domain_df) > 0:
        for key in ['activity', 'screen', 'pedo', 'light', 'w_light']:
            if key in daily:
                domain_cols = [c for c in domain_df.columns
                               if c.startswith(key[:3]) or
                               (key == 'light'   and c.startswith('mLight')) or
                               (key == 'w_light' and c.startswith('wLight'))]
                sub_domain = domain_df[['subject_id', 'date'] + domain_cols].dropna(
                    subset=domain_cols, how='all')
                if len(sub_domain.columns) > 2:
                    daily[key] = pd.merge(daily[key], sub_domain,
                                          on=['subject_id', 'date'], how='outer')
        hr_domain_cols = [c for c in domain_df.columns if c.startswith('hr_')]
        if hr_domain_cols:
            daily['hr_domain'] = domain_df[['subject_id', 'date'] + hr_domain_cols]

    print('📊 [3] 중첩 센서 — HR')
    if 'hr' in raw:
        hr_base   = extract_hr_features_nested(raw['hr'])
        hr_sleep  = estimate_sleep_from_hr(raw['hr'])
        hr_merged = pd.merge(hr_base, hr_sleep, on=['subject_id', 'date'], how='outer')
        if 'hr_domain' in daily:
            hr_merged = pd.merge(hr_merged, daily.pop('hr_domain'),
                                 on=['subject_id', 'date'], how='outer')
        daily['hr'] = hr_merged

    print('📊 [4] 중첩 센서 — WiFi')
    if 'wifi' in raw:
        wifi_base = extract_wifi_features_nested(raw['wifi'])
        wifi_bed  = extract_wifi_bedtime_proxy(raw['wifi'])
        daily['wifi'] = pd.merge(wifi_base, wifi_bed,
                                 on=['subject_id', 'date'], how='outer')

    print('📊 [5] 중첩 센서 — UsageStats')
    if 'usage_stats' in raw:
        usage_base = extract_usage_stats_features_nested(raw['usage_stats'])
        usage_bed  = extract_usage_bedtime_proxy(raw['usage_stats'])
        daily['usage_stats'] = pd.merge(usage_base, usage_bed,
                                        on=['subject_id', 'date'], how='outer')

    print('📊 [6] 중첩 센서 — GPS / Ambience / BLE')
    if 'gps'      in raw: daily['gps']      = extract_gps_special_features(raw['gps'])
    if 'ambience' in raw: daily['ambience'] = extract_ambience_special_features(raw['ambience'])
    if 'ble'      in raw: daily['ble']      = extract_ble_special_features(raw['ble'])

    print('📊 [7] 전체 센서 merge')
    feature_df = reduce(
        lambda l, r: pd.merge(l, r, on=['subject_id', 'date'], how='outer'),
        daily.values()
    )

    n_feat = feature_df.shape[1] - 2
    print(f'✅ 통합 피처 완료: {n_feat}개 피처 / {len(feature_df)}행')
    print(f'   포함 센서: {list(daily.keys())}')
    return feature_df