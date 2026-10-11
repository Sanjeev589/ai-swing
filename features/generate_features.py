"""Generate separate 2019–2022 feature files from validated official inputs."""
import argparse
import lzma
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features.swing_indicators import calculate_features


def generate(inputs='data/feature_inputs', output='features/derived', start_year=2019, end_year=2022):
    inputs, output = Path(inputs), Path(output)
    paths = sorted(inputs.glob('ohlcv_*.csv.gz'))
    if not paths or start_year > end_year: raise ValueError('Missing inputs or invalid years')
    prices = pd.concat([pd.read_csv(path, parse_dates=['date']) for path in paths], ignore_index=True)
    benchmark = pd.read_csv(inputs/'nifty500_index.csv', parse_dates=['date'])
    features = calculate_features(prices, benchmark)
    features = features[features.date.dt.year.between(start_year,end_year)].copy()
    numeric = features.select_dtypes(include='number')
    if np.isinf(numeric.to_numpy()).any() or features.duplicated(['date','symbol']).any():
        raise ValueError('Invalid feature output')
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for year in range(start_year,end_year+1):
        annual = features[features.date.dt.year.eq(year)]
        if annual.empty: raise ValueError(f'No observations for {year}')
        path = output/f'features_{year}.csv.xz'
        path.write_bytes(lzma.compress(annual.to_csv(index=False, date_format='%Y-%m-%d',float_format='%.10g').encode(),preset=6))
        files.append({'path':path.name,'rows':len(annual),'symbols':annual.symbol.nunique(),
                      'sessions':annual.date.nunique(),'ready_rows':int(annual.features_ready.sum()),
                      'missing_by_column':{k:int(v) for k,v in annual.isna().sum().items()},
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'years':[start_year,end_year],'columns':list(features.columns),'files':files,
                'inputs_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[inputs/'nifty500_index.csv']},
                'causality':'Indicators use current and earlier sessions only; available after session close',
                'prices':'Unadjusted NSE prices; current constituent snapshot; no historical membership reconstruction',
                'warmup':'2017-12-01 onward; no future fills; missing/invalid bars reset indicators',
                'relative_strength':'Stock gross return / Nifty 500 price-index gross return minus 1; exact 20/60 benchmark sessions',
                'high_proximity':'252-session high; ratio and percentage distance; minimum 252 contiguous valid bars'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(files,indent=2))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',default='data/feature_inputs'); parser.add_argument('--output',default='features/derived')
    parser.add_argument('--start-year',type=int,default=2019); parser.add_argument('--end-year',type=int,default=2022)
    generate(**vars(parser.parse_args()))
