"""Official NSE daily equity bhavcopy importer and non-destructive Yahoo comparison.

Uses user-supplied ZIP/CSV or optionally a public NSE archive URL. No account,
web scraping, broker credentials, or live-trade execution. NSE OHLCV is RAW,
never substituted for Yahoo's provider-adjusted historical series.
"""
from __future__ import annotations

import argparse
from datetime import date
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from marketdata.shared_store import bar_path, data_root, load_universe, save_year, validate_bars, validate_universe_name

ARCHIVE = 'https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv.zip'

ALIASES = {
    'symbol':['TckrSymb','SYMBOL','Symbol'],
    'isin':['ISIN','ISIN_CODE','ISINCode'],
    'series':['SctySrs','SERIES','Series'],
    'date':['TradDt','DATE1','TIMESTAMP','DATE'],
    'open':['OpnPric','OPEN_PRICE','OPEN'],
    'high':['HghPric','HIGH_PRICE','HIGH'],
    'low':['LwPric','LOW_PRICE','LOW'],
    'close':['ClsPric','CLOSE_PRICE','CLOSE'],
    'volume':['TtlTradgVol','TTL_TRD_QNTY','TOTTRDQTY'],
}


def _extract_table(payload: bytes):
    if zipfile.is_zipfile(BytesIO(payload)):
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names=[n for n in archive.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX')]
            if len(names)!=1: raise ValueError('Expected exactly one CSV in NSE archive')
            payload=archive.read(names[0])
    return pd.read_csv(BytesIO(payload),dtype=str,encoding='utf-8-sig',low_memory=False)


def parse_nse_bhavcopy(payload: bytes, *, session: str, universe: pd.DataFrame) -> pd.DataFrame:
    frame=_extract_table(payload)
    frame.columns=[c.strip() for c in frame.columns]
    column={}
    for key,options in ALIASES.items():
        col=next((k for k in options if k in frame.columns),None)
        if col is None and key not in {'isin','series'}:
            raise ValueError(f'Unsupported NSE bhavcopy: missing {key}')
        if col is not None:column[key]=col
    day=pd.Timestamp(session)
    if day != day.normalize(): raise ValueError('Expected daily session date')
    if 'series' in column:
        frame=frame[frame[column['series']].str.strip().eq('EQ')].copy()
    frame['symbol']=frame[column['symbol']].str.strip().str.upper()
    frame=frame[frame.symbol.isin(set(universe.symbol))].copy()
    if frame.empty: raise ValueError('No matching equity bars in NSE file')
    # Historical NSE files vary between ISO and DD-Mon-YYYY formats.
    datevalues=pd.to_datetime(frame[column['date']].str.strip(),errors='coerce',format='mixed',dayfirst=True)
    if datevalues.isna().any() or not datevalues.eq(day).all():
        raise ValueError('NSE archive session does not match requested date')
    frame['date']=day
    identifiers=universe.set_index('symbol')['isin']
    frame['isin']=frame.symbol.map(identifiers)
    if 'isin' in column:
        provided=frame[column['isin']].str.strip().str.upper()
        if (provided.ne('') & provided.ne(frame['isin'])).any():
            raise ValueError('ISIN mismatch between current universe and NSE archive')
    for key in ('open','high','low','close','volume'):
        frame[key]=pd.to_numeric(frame[column[key]].astype(str).str.replace(',','',regex=False),errors='raise')
    if frame.symbol.duplicated().any():
        raise ValueError('Duplicate symbol in NSE daily archive')
    return frame[list(('date','symbol','isin','open','high','low','close','volume'))].sort_values('symbol').reset_index(drop=True)


def _download_archive(session: str, *, http=None, attempts=3):
    http=http or requests.Session()
    url=ARCHIVE.format(stamp=date.fromisoformat(session).strftime('%Y%m%d'))
    for n in range(attempts):
        try:
            response=http.get(url,timeout=(10,35),headers={'User-Agent':'Mozilla/5.0','Accept':'application/zip'})
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if n==attempts-1:raise
            time.sleep(2**n)
    raise AssertionError('unreachable')


def import_day(*, root=None, snapshot, session, archive_file=None, http=None,
               universe_name='nifty50'):
    """Import one published day. Date/identity mismatch fails; no adjusted-price mutation."""
    root=data_root(root)
    universe=load_universe(root,snapshot,universe_name=universe_name)
    date.fromisoformat(session)
    if archive_file:
        payload=Path(archive_file).read_bytes()
        archive_source=str(archive_file)
    else:
        payload=_download_archive(session,http=http)
        archive_source=ARCHIVE.format(stamp=session.replace('-',''))
    frame=parse_nse_bhavcopy(payload,session=session,universe=universe)
    # Validate per ISIN; validate_bars deliberately rejects mixed-symbol partitions.
    for _, stock in frame.groupby('symbol'):
        validate_bars(stock)
    year=int(session[:4]);updated=0
    for item in frame.itertuples(index=False):
        path=bar_path(root,item.isin,year,'nse_raw')
        next_day=pd.DataFrame([item._asdict()])
        if path.exists():
            old=validate_bars(pd.read_csv(path,dtype={'isin':str}))
            conflict=old[old.date.eq(pd.Timestamp(session))]
            if not conflict.empty:
                # Exact identical re-import is idempotent; changes are never silent.
                a=conflict.iloc[0];b=next_day.iloc[0]
                if any(a[k] != b[k] for k in ('symbol','isin','open','high','low','close','volume')):
                    raise ValueError(f'Existing NSE raw prices changed for {item.symbol} {session}')
                continue
            next_day=pd.concat([old,next_day],ignore_index=True)
        save_year(root,next_day,isin=item.isin,year=year,provider='nse_raw',replace=path.exists())
        updated+=1
    # Compare same-day OHLC only with a clear corporate-action warning. No auto-merge.
    comparisons=[]
    for x in frame.itertuples(index=False):
        yahoo=bar_path(root,x.isin,year,'yahoo_finance')
        if not yahoo.is_file():continue
        yahoo_rows=validate_bars(pd.read_csv(yahoo,dtype={'isin':str}))
        match=yahoo_rows[yahoo_rows.date.eq(pd.Timestamp(session))]
        if match.empty:continue
        previous=match.iloc[0]
        comparisons.append({'date':session,'symbol':x.symbol,'isin':x.isin,
            'yahoo_close_adjusted':float(previous.close),'nse_close_raw':float(x.close),
            'raw_vs_adjusted_close_diff_pct':float(100*(previous.close/x.close-1)),
            'review_required':abs(previous.close/x.close-1)>.02})
    reports=root/'reports'/'nse_raw';reports.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(comparisons,columns=['date','symbol','isin','yahoo_close_adjusted','nse_close_raw',
                       'raw_vs_adjusted_close_diff_pct','review_required']).to_csv(
        reports/f'{universe_name}_{snapshot}_{session}_comparisons.csv',index=False)
    summary={'date':session,'snapshot':snapshot,'provider':'nse_raw','universe_name':universe_name,
        'official_source':archive_source,'stocks_in_daily_archive':int(len(frame)),
        'new_or_updated_partitions':updated,'yahoo_overlap_rows':len(comparisons),
        'important':'NSE is unadjusted; Yahoo OHLC are adjusted. Do not treat price differences as execution errors or overwrite Yahoo data.',
        'live_execution_enabled':False}
    (reports/f'{universe_name}_{snapshot}_{session}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root')
    p.add_argument('--snapshot',required=True)
    p.add_argument('--universe-name',default='nifty50')
    p.add_argument('--date',required=True,help='NSE trading session YYYY-MM-DD')
    p.add_argument('--archive',help='Locally downloaded official NSE ZIP/CSV; omit to attempt official URL')
    a=p.parse_args()
    print(json.dumps(import_day(root=a.data_root,snapshot=a.snapshot,session=a.date,archive_file=a.archive,universe_name=a.universe_name),indent=2))

if __name__=='__main__': main()
